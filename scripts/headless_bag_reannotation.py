"""Headless per-bag super_odometry reannotation for OSMO"""
import argparse
import os
import shutil
import signal
import subprocess
import time

import yaml

# Defaults for the playback config's `reannotation:` block (see load_reannotation_settings).
REANNOTATION_DEFAULTS = {
    "reannotated_dir": "",
    "record_drain": 3.0,
    "force": False,
    "remap_prefix": "/prereannotation",
    "keep_remapped": True,
    # Recorded topics the original bag legitimately has too, so they can neither gate
    # reannotation nor be remapped away from the live stack.
    "shared_topics": ["/tf", "/tf_static"],
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.environ.get("TARTANDRIVER_HOME") or \
    os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_DEPLOY_DIR = os.path.join(_REPO_ROOT, "tartandriver_deploy")

DEFAULT_CONFIG = os.path.join(
    _DEPLOY_DIR, "configs", "playback", "postprocess_super_odometry.yaml")
DEFAULT_REGISTRY = os.path.join(_DEPLOY_DIR, "default_registry.yaml")


def resolve_deploy_path(path):
    """Resolve `path` against the repo root ($TARTANDRIVER_HOME) if it isn't already absolute."""
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def resolve_global_parameters_path(path):
    """Locate a config's `global_parameters_fp` in this checkout or fall back to re-rooting the
    `tartandriver_deploy/...` or None"""
    candidate = resolve_deploy_path(path)
    if os.path.exists(candidate):
        return candidate
    marker = "tartandriver_deploy" + os.sep
    idx = path.find(marker)
    if idx != -1:
        candidate = os.path.join(_REPO_ROOT, path[idx:])
        if os.path.exists(candidate):
            return candidate
    return None


def launch_arg(key, value):
    """`key:=value` for ros2 launch, with YAML bools lowercased to ROS spelling."""
    if isinstance(value, bool):
        value = str(value).lower()
    return "{}:={}".format(key, value)


def default_models_dir():
    """models_dir launch arg derived from $MODELS_DIR or $TARTANDRIVER_HOME/models."""
    if os.environ.get("MODELS_DIR"):
        return os.environ["MODELS_DIR"]
    home = os.environ.get("TARTANDRIVER_HOME")
    return os.path.join(home, "models") if home else ""


def build_stack_from_config(config_path, registry_path, models_dir="", use_sim_time=True):
    """Derive launches/remap/record settings from a tartandriver_deploy config + registry"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    global_params = {}
    global_fp = config.get("global_parameters_fp")
    if global_fp:
        global_path = resolve_global_parameters_path(global_fp)
        if global_path:
            with open(global_path, "r") as f:
                global_params = yaml.safe_load(f) or {}
        else:
            print("[reannotate] global_parameters_fp '{}' not found, "
                  "using use_sim_time/models_dir defaults only".format(global_fp), flush=True)

    extra = [launch_arg(k, v) for k, v in global_params.items()
             if k not in ("use_sim_time", "models_dir")]
    if use_sim_time:
        extra.append("use_sim_time:=true")
    if models_dir:
        extra.append("models_dir:={}".format(models_dir))

    launches = []
    for key, spec in (config.get("launch") or {}).items():
        assert key in registry.get("launch", {}), \
            "launch '{}' in {} not found in registry {}".format(key, config_path, registry_path)
        cmd = registry["launch"][key]["launch_cmd"].split()
        for k, v in (spec.get("launch_args") or {}).items():
            cmd.append(launch_arg(k, v))
        # `provides:` lets reannotate_bag skip a component whose output the bag already
        # carries; absent means always launch.
        launches.append({"key": key, "cmd": cmd + extra,
                         "provides": list(spec.get("provides") or []),
                         "provides_shared": list(spec.get("provides_shared") or [])})

    playback_args = (config.get("rosbag_playback", {}) or {}).get("args", {}) or {}
    remap = (playback_args.get("--remap", "") or "").split()
    play_rate = float(playback_args.get("--rate", 1.0))
    start_offset = playback_args.get("--start-offset", None)

    handled = {"--rate", "--start-offset", "--remap"}
    extra_play_args = []
    for k, v in playback_args.items():
        if k in handled:
            continue
        extra_play_args.append(str(k))
        if v not in (None, ""):
            extra_play_args.append(str(v))

    record = config.get("rosbag_record", {}) or {}
    record_storage = (record.get("args", {}) or {}).get("-s", "mcap")

    return {
        "launches": launches,
        "remap": remap,
        "record_topics": record.get("topics", "all"),
        "record_storage": record_storage,
        "play_rate": play_rate,
        "start_offset": None if start_offset in (None, "") else str(start_offset),
        "extra_play_args": extra_play_args,
    }


def bag_topics(bag_dir):
    """Return {topic_name: message_count} parsed from a run-dir's metadata.yaml."""
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    with open(meta_path, "r") as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict) or "rosbag2_bagfile_information" not in metadata:
        raise ValueError(
            f"{meta_path} is empty or malformed (no 'rosbag2_bagfile_information') -- "
            "the recording likely never shut down cleanly, so this bag is unusable"
        )
    info = metadata["rosbag2_bagfile_information"]
    return {
        t["topic_metadata"]["name"]: t["message_count"]
        for t in info["topics_with_message_count"]
    }


def present_topics(bag_dir, topics):
    """Subset of `topics` the bag actually carries (present with a non-zero count)."""
    counts = bag_topics(bag_dir)
    return [t for t in topics if counts.get(t, 0) > 0]


def needed_launches(launches_spec, src_dir, force=False):
    """Drop launches whose `provides:` topics the bag already carries.

    Playback republishes those topics on their original names, so anything downstream
    still gets its input without paying to regenerate it. `force` keeps every launch:
    it remaps the existing copies out of the way, so the live producer has to run.
    """
    if force:
        return list(launches_spec)
    kept = []
    for launch in launches_spec:
        provides = launch["provides"]
        if provides and len(present_topics(src_dir, provides)) == len(provides):
            print("[reannotate] skipping launch '{}': the bag already has {}".format(
                launch["key"], ", ".join(provides)), flush=True)
            continue
        kept.append(launch)
    return kept


def shared_to_record(all_launches, kept_launches, shared_topics):
    """Which `shared_topics` are worth recording, given which launches actually run.

    A launch declares the shared topics it writes with `provides_shared:` (e.g. super
    odometry writes /tf). Skip that launch and nothing regenerates those, so recording
    them would only capture the playback echo of what the bag already has. If no launch
    in the config declares any, every shared topic is recorded, as before.
    """
    declared = {t for launch in all_launches for t in launch["provides_shared"]}
    if not declared:
        return set(shared_topics)
    running = {t for launch in kept_launches for t in launch["provides_shared"]}
    dropped = sorted(set(shared_topics) & declared - running)
    if dropped:
        print("[reannotate] not recording {}: nothing running regenerates them".format(
            ", ".join(dropped)), flush=True)
    return set(shared_topics) & (running | (set(shared_topics) - declared))


def bag_has_topics(bag_dir, topics):
    """True iff every topic in `topics` is in the bag with a non-zero message count."""
    topics = list(topics)
    return len(present_topics(bag_dir, topics)) == len(topics)


def load_reannotation_settings(config_path):
    """`reannotation:` block of a playback config, with REANNOTATION_DEFAULTS filled in.

    Read by both this module and dataset_pipeline.py, so the block stays the one
    place these knobs are configured."""
    with open(config_path, "r") as f:
        section = (yaml.safe_load(f) or {}).get("reannotation") or {}
    settings = dict(REANNOTATION_DEFAULTS)
    settings.update({k: v for k, v in section.items() if v is not None})
    settings["reannotated_dir"] = str(settings["reannotated_dir"] or "")
    settings["record_drain"] = float(settings["record_drain"])
    settings["force"] = bool(settings["force"])
    settings["keep_remapped"] = bool(settings["keep_remapped"])
    settings["shared_topics"] = list(settings["shared_topics"])
    return settings


def regenerated_topics(record_topics, shared_topics):
    """The subset of `rosbag_record.topics` that only the annotation stack produces"""
    if record_topics == "all":
        return []
    shared = set(shared_topics)
    return [t for t in record_topics if t not in shared]


def config_regenerated_topics(config_path, settings=None):
    """regenerated_topics() for a playback config's `rosbag_record.topics`."""
    settings = settings or load_reannotation_settings(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    record_topics = (config.get("rosbag_record") or {}).get("topics", "all")
    return regenerated_topics(record_topics, settings["shared_topics"])


def remapped_topic(topic, prefix):
    """`topic` moved under `prefix` (e.g. /superodometry/x -> /prereannotation/superodometry/x)."""
    return "{}/{}".format(str(prefix).rstrip("/"), topic.lstrip("/"))


def _sigint_group(proc):
    """SIGINT a process group so ros2 launch/record shut down cleanly (mcap flushed)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        pass


def _wait_group(proc, grace):
    """Wait up to `grace` s for a process to exit, then SIGKILL its group."""
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def reannotate_bag(src_dir, out_dir, domain_id=None,
                   config_path=DEFAULT_CONFIG, registry_path=DEFAULT_REGISTRY,
                   models_dir=None, use_sim_time=True, settle=8.0,
                   record_settle=2.0, shutdown_grace=15.0,
                   localhost_only=True, settings=None, record_only=None):
    """Reannotate a single bag headlessly; returns the resulting bag path (merged with
    the untouched original when `rosbag_record.topics` is a list of topics).

    `record_only` narrows `rosbag_record.topics` to the topics this bag is actually
    missing (plus `shared_topics`, which the stack may add to). Without it, playback
    republishes the copies the bag already has, the recorder picks those up, and the
    merge leaves two of everything. None = record the whole configured list, which is
    what `force` wants."""
    if models_dir is None:
        models_dir = default_models_dir()
    settings = settings or load_reannotation_settings(config_path)
    spec = build_stack_from_config(config_path, registry_path,
                                   models_dir=models_dir, use_sim_time=use_sim_time)
    launches = needed_launches(launches_spec=spec["launches"], src_dir=src_dir,
                               force=settings["force"])
    remap = list(spec["remap"])
    record_topics = spec["record_topics"]
    record_storage = spec["record_storage"]
    play_rate = spec["play_rate"]
    start_offset = spec["start_offset"]
    extra_play_args = spec.get("extra_play_args", [])
    record_drain = settings["record_drain"]

    if record_only is not None and record_topics != "all":
        keep = set(record_only) | shared_to_record(spec["launches"], launches,
                                                   settings["shared_topics"])
        skipped = [t for t in record_topics if t not in keep]
        record_topics = [t for t in record_topics if t in keep]
        if skipped:
            print("[reannotate] recording only {} -- the bag already has {}, left "
                  "untouched".format(", ".join(record_topics), ", ".join(skipped)),
                  flush=True)

    remapped = {}
    if settings["force"]:
        regenerated = regenerated_topics(record_topics, settings["shared_topics"])
        remapped = {t: remapped_topic(t, settings["remap_prefix"])
                    for t in present_topics(src_dir, regenerated)}
    if remapped:
        remap += ["{}:={}".format(old, new) for old, new in sorted(remapped.items())]
        print("[reannotate] force: replaying existing {} under {}".format(
            ", ".join(sorted(remapped)), settings["remap_prefix"]), flush=True)
        if settings["keep_remapped"] and record_topics != "all":
            record_topics = list(record_topics) + [remapped[t] for t in sorted(remapped)]

    env = os.environ.copy()
    if domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(domain_id)
    if localhost_only:
        # Confines DDS discovery to loopback so this graph can't see/be seen by any other on the node.
        env["ROS_LOCALHOST_ONLY"] = "1"

    os.makedirs(out_dir, exist_ok=True)
    bag_name = os.path.basename(os.path.normpath(src_dir))
    # 'all' is the legacy no-merge path; a topic list triggers merge_bags() below.
    merge_new_topics = record_topics != "all"
    record_path = os.path.join(out_dir, bag_name + "_new_topics") if merge_new_topics \
        else os.path.join(out_dir, bag_name)

    procs = []

    def start(cmd):
        # New session/process group so we can signal the whole subtree on teardown.
        p = subprocess.Popen(cmd, env=env, start_new_session=True)
        procs.append(p)
        return p

    try:
        # 1. bring up the annotation stack
        for launch in launches:
            start(launch["cmd"])
        time.sleep(settle)

        # 2. record; 'all' -> -a since $ALL_TOPICS isn't available in a headless pod.
        record_cmd = ["ros2", "bag", "record", "--use-sim-time"]
        if record_topics == "all":
            record_cmd.append("-a")
        else:
            record_cmd += list(record_topics)
        record_cmd += ["-s", record_storage, "-o", record_path]
        record_proc = start(record_cmd)
        time.sleep(record_settle)  # let the recorder subscribe before data flows

        # 3. play the source bag; blocks until it finishes.
        # --clock + --use-sim-time keep new topics on bag time, aligned for the merge.
        play_cmd = ["ros2", "bag", "play", src_dir, "--clock", "--rate", str(play_rate)]
        if start_offset is not None:
            play_cmd += ["--start-offset", str(start_offset)]
        play_cmd += list(extra_play_args)
        if remap:
            play_cmd += ["--remap"] + list(remap)  # --remap consumes the rest, keep it last
        subprocess.run(play_cmd, env=env, check=True)
        time.sleep(record_drain)

        # 4. stop the recorder first so it flushes a complete mcap
        _sigint_group(record_proc)
        _wait_group(record_proc, shutdown_grace)

        if not merge_new_topics:
            return record_path
        merge_src, filtered_dir = src_dir, None
        try:
            if remapped:
                filtered_dir = os.path.join(out_dir, bag_name + "_orig_filtered")
                merge_src = filter_bag(src_dir, filtered_dir, list(remapped),
                                       storage_id=record_storage, env=env)
            return merge_bags(merge_src, record_path, out_dir, bag_name,
                              storage_id=record_storage, env=env)
        finally:
            if filtered_dir:
                shutil.rmtree(filtered_dir, ignore_errors=True)
    finally:
        # 5. tear the whole stack down (reverse order); record already handled above
        for p in reversed(procs):
            _sigint_group(p)
        for p in reversed(procs):
            _wait_group(p, shutdown_grace)


def filter_bag(src_dir, out_path, drop_topics, storage_id="mcap", env=None):
    """Copy `src_dir` to `out_path` minus `drop_topics`, via `ros2 bag convert`"""
    drop = set(drop_topics)
    keep = [t for t in bag_topics(src_dir) if t not in drop]
    assert keep, "dropping {} would leave nothing in {}".format(sorted(drop), src_dir)
    convert_opts_path = out_path + "_convert_opts.yaml"
    with open(convert_opts_path, "w") as f:
        yaml.safe_dump({"output_bags": [
            {"uri": out_path, "storage_id": storage_id, "topics": keep},
        ]}, f)
    subprocess.run(
        ["ros2", "bag", "convert", "-i", src_dir, "-o", convert_opts_path],
        env=env, check=True,
    )
    return out_path


def merge_bags(bag_a, bag_b, out_dir, out_name, storage_id="mcap", env=None):
    """Combine two bag dirs losslessly into `out_dir/out_name` via `ros2 bag convert`
    (copies messages as-is, no live graph -- safe when originals must stay untouched)."""
    merged_path = os.path.join(out_dir, out_name)
    convert_opts_path = os.path.join(out_dir, out_name + "_convert_opts.yaml")
    with open(convert_opts_path, "w") as f:
        yaml.safe_dump({"output_bags": [
            {"uri": merged_path, "storage_id": storage_id, "all": True},
        ]}, f)
    subprocess.run(
        ["ros2", "bag", "convert", "-i", bag_a, "-i", bag_b, "-o", convert_opts_path],
        env=env, check=True,
    )
    return merged_path
