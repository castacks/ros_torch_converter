"""Headless per-bag super_odometry reannotation for OSMO"""
import argparse
import os
import signal
import subprocess
import time

import yaml

SO_TOPICS = [
    "/superodometry/integrated_to_init",
    "/superodometry/velodyne_cloud_registered",
]

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

    extra = []
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
            cmd.append("{}:={}".format(k, v))
        launches.append(cmd + extra)

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


def bag_has_superodometry(bag_dir):
    """True iff both SO topics are present in the bag with a non-zero message count."""
    topics = bag_topics(bag_dir)
    return all(topics.get(t, 0) > 0 for t in SO_TOPICS)


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
                   record_settle=2.0, record_drain=3.0, shutdown_grace=15.0,
                   localhost_only=True):
    """Reannotate a single bag headlessly; returns the resulting bag path (merged with
    the untouched original when `rosbag_record.topics` is a list of topics)."""
    if models_dir is None:
        models_dir = default_models_dir()
    spec = build_stack_from_config(config_path, registry_path,
                                   models_dir=models_dir, use_sim_time=use_sim_time)
    launches = spec["launches"]
    remap = spec["remap"]
    record_topics = spec["record_topics"]
    record_storage = spec["record_storage"]
    play_rate = spec["play_rate"]
    start_offset = spec["start_offset"]
    extra_play_args = spec.get("extra_play_args", [])

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
            start(launch)
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
        return merge_bags(src_dir, record_path, out_dir, bag_name,
                          storage_id=record_storage, env=env)
    finally:
        # 5. tear the whole stack down (reverse order); record already handled above
        for p in reversed(procs):
            _sigint_group(p)
        for p in reversed(procs):
            _wait_group(p, shutdown_grace)


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src_dir", required=True, help="raw rosbag run-dir to reannotate")
    parser.add_argument("--out_dir", required=True, help="dir to write the reannotated bag into")
    parser.add_argument("--domain_id", type=int, default=None, help="ROS_DOMAIN_ID for this run")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="tartandriver_deploy playback config defining the stack")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help="tartandriver_deploy launch registry")
    parser.add_argument("--models_dir", default=default_models_dir(),
                        help="models_dir launch arg (default $MODELS_DIR or $TARTANDRIVER_HOME/models)")
    parser.add_argument("--settle", type=float, default=8.0)
    parser.add_argument("--force", action="store_true",
                        help="reannotate even if the bag already has super_odometry")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.force and bag_has_superodometry(args.src_dir):
        print("[reannotate] {} already has super_odometry; skipping".format(args.src_dir))
        return
    out = reannotate_bag(
        args.src_dir, args.out_dir, domain_id=args.domain_id,
        config_path=args.config, registry_path=args.registry,
        models_dir=args.models_dir, settle=args.settle,
    )
    print("[reannotate] wrote {}".format(out))


if __name__ == "__main__":
    main()
