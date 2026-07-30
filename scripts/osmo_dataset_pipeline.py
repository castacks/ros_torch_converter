"""OSMO dataset discovery + conversion pipeline driver.
Runs two phases per batch of pending bags:
  1. reannotate
  2. convert
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from tartandriver_utils.os_utils import available_cpus, is_kitti_dir, is_rosbag_dir_filenames, load_yaml
from tartandriver_utils.rclone_stager import RcloneStager

from headless_bag_reannotation import bag_has_superodometry, reannotate_bag, resolve_deploy_path
from bag_message_report import build_report, write_report
from topic_sync_viz import render_sync_viz


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERTER_SCRIPT = os.path.join(SCRIPT_DIR, "ros2bag_2_kitti_multiproc.py")
CONVERTER_PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)


def resolve_config_path(path):
    """Resolve `path` against the ros_torch_converter package dir if it isn't already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(CONVERTER_PACKAGE_DIR, path)


def _init_worker_stdio():
    """ProcessPoolExecutor initializer: keep worker stdout/stderr line-buffered so
    per-bag progress lands in the OSMO log as it happens"""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


def chunked(seq, size):
    """Split `seq` into consecutive chunks of `size`; one chunk of everything if size<=0."""
    seq = list(seq)
    if not size or size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def start_memory_logger(scratch_root, interval=60.0):
    """Log cgroup memory use + scratch-disk free space periodically, as a daemon thread.
    """
    def read(name):
        for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
            try:
                with open(os.path.join(base, name)) as f:
                    return int(f.read().split()[0])
            except (OSError, ValueError):
                continue
        return None

    # cgroup v2 names first, then v1 equivalents.
    limit = read("memory.max") or read("memory.limit_in_bytes")
    mem_available = read("memory.current") is not None or read("memory.usage_in_bytes") is not None
    if not mem_available:
        print("[mem] cgroup memory stats unavailable -- skipping memory logging")

    gib = 1024 ** 3
    limit_str = f"{limit / gib:.1f}Gi" if limit else "unlimited"
    stop = threading.Event()

    def poll():
        peak = 0
        while not stop.is_set():
            if mem_available:
                cur = read("memory.current") or read("memory.usage_in_bytes") or 0
                peak = max(peak, cur)
                pct = f" ({100 * cur / limit:.0f}% of limit)" if limit else ""
                print(f"[mem] {cur / gib:.1f}Gi in use, peak {peak / gib:.1f}Gi, "
                      f"limit {limit_str}{pct}", flush=True)
            try:
                free = shutil.disk_usage(scratch_root).free
                print(f"[disk] {free / gib:.1f}Gi free on {scratch_root}", flush=True)
            except OSError:
                pass
            stop.wait(interval)  # wake immediately on stop instead of sleeping out the interval

    threading.Thread(target=poll, daemon=True).start()
    return stop.set  # call to stop logging (it's a daemon, so this is just to quiet the log)


def copy_bag_metadata(orig_dir, dst_dir):
    src = os.path.join(orig_dir, "info.yaml")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, "info.yaml"))


def local_lsdir_recursive(root_dir):
    """Local-filesystem equivalent of RcloneStager.list(), for full-copy (data_dir) mode."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        relpath = os.path.relpath(dirpath, root_dir)
        for name in filenames:
            entries.append({"IsDir": False, "Path": os.path.normpath(os.path.join(relpath, name))})
    return entries


def find_rosbag_run_dirs(root_dir, exclude_subdirs, limit_subfolder=None, stager=None, data_dir=None):
    """Find run-dirs under root_dir with metadata.yaml + >=1 .mcap, at any depth."""
    entries = stager.list(root_dir) if stager else local_lsdir_recursive(os.path.join(data_dir, root_dir))

    filenames_by_dir = {}
    for entry in entries:
        if entry["IsDir"]:
            continue
        parent = os.path.dirname(entry["Path"])
        filenames_by_dir.setdefault(parent, []).append(os.path.basename(entry["Path"]))

    run_dirs = []
    for relpath, filenames in filenames_by_dir.items():
        if limit_subfolder is not None and relpath != limit_subfolder:
            continue
        if any(part in exclude_subdirs for part in relpath.split("/")):
            continue
        if is_rosbag_dir_filenames(filenames):
            run_dirs.append(relpath)

    return sorted(run_dirs)


def filter_unconverted(run_dirs, dst_dir, stager=None, data_dir=None):
    """Drop run-dirs that already have a completed KITTI dataset at the destination."""
    if stager:
        return [relpath for relpath in run_dirs
                if not stager.exists(os.path.join(dst_dir, relpath))]
    return [relpath for relpath in run_dirs
            if not is_kitti_dir(os.path.join(data_dir, dst_dir, relpath))]


def classify_reannotation_needed(relpaths, root_dir, stager, data_dir):
    """Subset of `relpaths` whose raw bag is missing super_odometry, checked via
    metadata.yaml only (no full-bag download) so phase 1 only stages bags it will
    actually reannotate."""
    needed = []
    for relpath in relpaths:
        if stager:
            tmpdir = tempfile.mkdtemp(prefix="osmo_pipeline_meta_")
            try:
                stager.copy_in(os.path.join(root_dir, relpath, "metadata.yaml"), tmpdir)
                if not bag_has_superodometry(tmpdir):
                    needed.append(relpath)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            bag_dir = os.path.join(data_dir, root_dir, relpath)
            if not bag_has_superodometry(bag_dir):
                needed.append(relpath)
    return needed


def reannotate_one(relpath, root_dir, kitti_out_root, orig_scratch_root, merged_scratch_root,
                   dst_dir, pipeline_config, reannotate_config, reannotated_dir, record_drain,
                   domain_queue, stager, data_dir):
    domain_id = None
    try:
        if stager:
            orig_dir = os.path.join(orig_scratch_root, relpath)
            os.makedirs(orig_dir, exist_ok=True)
            print(f"[reannotate] {relpath}: copying raw bag from {stager.remote} ...", flush=True)
            stager.copy_in(os.path.join(root_dir, relpath), orig_dir)
        else:
            orig_dir = os.path.join(data_dir, root_dir, relpath)

        if domain_queue is not None:
            domain_id = domain_queue.get()
        print(f"[reannotate] {relpath}: missing super_odometry -- reannotating "
              f"(ROS_DOMAIN_ID={domain_id}) ...", flush=True)

        merged_root = os.path.join(merged_scratch_root, relpath)
        reann_kwargs = {"config_path": reannotate_config} if reannotate_config else {}
        merged_dir = reannotate_bag(orig_dir, merged_root, domain_id=domain_id,
                                    record_drain=record_drain, **reann_kwargs)
        copy_bag_metadata(orig_dir, merged_dir)

        out_dir = os.path.join(kitti_out_root, relpath)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[reannotate] {relpath}: writing message-count report + sync viz ...", flush=True)
        report_rows = build_report(orig_dir, merged_dir)
        write_report(report_rows, os.path.join(out_dir, "reannotate_report"))
        try:
            render_sync_viz(merged_dir, resolve_config_path(pipeline_config),
                             os.path.join(out_dir, "topic_sync_viz.png"),
                             out_csv=os.path.join(out_dir, "topic_sync_viz.csv"))
        except Exception as e:
            print(f"[reannotate] {relpath}: topic_sync_viz failed (non-fatal): {e}", flush=True)

        if reannotated_dir:
            if stager:
                bag_dst_relpath = os.path.join(dst_dir, reannotated_dir, relpath)
                print(f"[reannotate] {relpath}: copying reannotated bag to "
                      f"{stager.remote}:{bag_dst_relpath} ...", flush=True)
                stager.copy_out(merged_dir, bag_dst_relpath)
            else:
                print(f"[reannotate] {relpath}: no stager configured -- skipping "
                      f"reannotated_dir save", flush=True)

        return relpath, True, None, merged_dir
    except Exception as e:
        return relpath, False, str(e), None
    finally:
        if domain_id is not None and domain_queue is not None:
            domain_queue.put(domain_id)  # release the domain for the next bag


def convert_one(relpath, root_dir, dst_dir, pipeline_config, converter_extra_args,
                render_video, video_config, kitti_out_root, stager, data_dir,
                local_out_dir, converter_workers, merged_bag_dir=None,
                orig_bag_dir=None, merged_scratch_dir=None):
    scratch_dir = tempfile.mkdtemp(prefix="osmo_pipeline_raw_")
    out_dir = os.path.join(kitti_out_root, relpath)
    os.makedirs(out_dir, exist_ok=True)

    try:
        if merged_bag_dir:
            print(f"[convert] {relpath}: using bag reannotated in phase 1 ...", flush=True)
            src_dir = merged_bag_dir
        elif stager:
            print(f"[convert] {relpath}: copying raw bag from {stager.remote} ...", flush=True)
            stager.copy_in(os.path.join(root_dir, relpath), scratch_dir)
            src_dir = scratch_dir
        else:
            src_dir = os.path.join(data_dir, root_dir, relpath)

        sync_viz_csv_path = os.path.join(out_dir, "topic_sync_viz.csv")
        if not os.path.exists(sync_viz_csv_path):  # phase 1 (reannotate) may have already written it
            try:
                render_sync_viz(src_dir, resolve_config_path(pipeline_config),
                                os.path.join(out_dir, "topic_sync_viz.png"),
                                out_csv=sync_viz_csv_path)
            except Exception as e:
                print(f"[convert] {relpath}: topic_sync_viz failed (non-fatal): {e}", flush=True)

        cmd = [
            sys.executable, CONVERTER_SCRIPT,
            "--config", resolve_config_path(pipeline_config),
            "--src_dir", src_dir,
            "--dst_dir", out_dir,
            "--force",
        ]
        if not render_video:
            cmd.append("--no_render_video")
        elif video_config:
            cmd.extend(["--video_config", resolve_config_path(video_config)])
        # Each converter sizes its own pool off the whole CPU budget, so N bags in
        # flight would together spawn N x that many frame-buffering workers. Split
        # the budget instead. An explicit --num_workers in converter_extra_args wins.
        if converter_workers and "--num_workers" not in converter_extra_args:
            cmd.extend(["--num_workers", str(converter_workers)])
        if converter_extra_args:
            cmd.extend(converter_extra_args.split())

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return relpath, False, f"conversion exited with code {rc}", None

        kept_frames = None
        try:
            with open(os.path.join(out_dir, "conversion_stats.json")) as f:
                kept_frames = json.load(f).get("kept_frames")
        except (OSError, json.JSONDecodeError):
            pass

        print(f"[convert] {relpath}: writing dataset report ...", flush=True)
        report_original_dir = orig_bag_dir if merged_bag_dir else src_dir
        copy_bag_metadata(report_original_dir, out_dir)
        report_rows = build_report(report_original_dir, merged_bag_dir, kitti_dir=out_dir)
        write_report(report_rows, os.path.join(out_dir, "dataset_report"))

        dst_relpath = os.path.join(dst_dir, relpath)
        if stager:
            print(f"[convert] {relpath}: copying result to {stager.remote}:{dst_relpath} ...", flush=True)
            stager.copy_out(out_dir, dst_relpath)
        else:
            local_dst = os.path.join(data_dir, dst_relpath)
            print(f"[convert] {relpath}: copying result to {local_dst} ...", flush=True)
            shutil.copytree(out_dir, local_dst, dirs_exist_ok=True)
        return relpath, True, None, kept_frames
    except subprocess.CalledProcessError as e:
        return relpath, False, str(e), None
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        # orig_bag_dir is only a scratch copy when a stager staged it (see
        # reannotate_one); in data_dir mode it's the caller's real source data.
        if orig_bag_dir and stager:
            shutil.rmtree(orig_bag_dir, ignore_errors=True)
        if merged_scratch_dir:
            shutil.rmtree(merged_scratch_dir, ignore_errors=True)
        if not local_out_dir:  # keep the dataset when a known output dir was requested
            shutil.rmtree(out_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="root to scan for run-dirs (rclone-remote-relative, or data_dir-relative if --data_dir is set)")
    parser.add_argument("--dst_dir", required=True, help="destination for the final KITTI copy-back (rclone-remote-relative, or data_dir-relative if --data_dir is set)")
    parser.add_argument("--data_dir", default="", help="local mount root for full-copy mode (bulk_copy.sh); when set, reads/writes go straight to this mount instead of through an RcloneStager")
    parser.add_argument("--pipeline_config", required=True,
                        help="path to a ros_torch_converter config yaml")
    parser.add_argument("--limit_subfolder", default=None, help="restrict discovery to a single run-dir relpath")
    parser.add_argument("--local_out_dir", default="", help="if set, write each converted KITTI dataset to <local_out_dir>/<relpath> and keep it there (instead of a scratch tempdir deleted after copy-back), so the caller can read results locally")
    parser.add_argument("--num_conversion_workers", type=int, default=None, help="bags to convert at once; overrides pipeline.num_conversion_workers")
    parser.add_argument("--num_reannotation_workers", type=int, default=None, help="bags to reannotate at once; overrides pipeline.num_reannotation_workers (default 1 -- one bag at a time so each live SLAM run gets the whole machine/GPU)")
    parser.add_argument("--phase_batch_size", type=int, default=None, help="bags per reannotate-then-convert batch; overrides pipeline.phase_batch_size (0/unset = one batch of everything pending)")
    parser.add_argument("--max_total_converter_workers", type=int, default=None, help="cap on converter pool workers across all bags; overrides pipeline.max_total_converter_workers (0 = use the cpu budget)")
    parser.add_argument("--converter_extra_args", default=None, help="extra args passed through to the converter; overrides pipeline.converter_extra_args")
    return parser.parse_args()


def pick(cli_value, cfg_value, name):
    """CLI (env-driven) value if given, else the pipeline-config value."""
    if cli_value is None:
        return cfg_value
    print(f"[config] {name}={cli_value} (overriding pipeline config value {cfg_value!r})")
    return cli_value


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    stager = RcloneStager() if not args.data_dir else None

    cfg = load_yaml(resolve_config_path(args.pipeline_config))
    pcfg = cfg.get("pipeline", {}) or {}

    num_conversion_workers = int(pick(args.num_conversion_workers,
                                      pcfg.get("num_conversion_workers", 4),
                                      "num_conversion_workers"))
    num_reannotation_workers = int(pick(args.num_reannotation_workers,
                                        pcfg.get("num_reannotation_workers", 1),
                                        "num_reannotation_workers"))
    phase_batch_size = int(pick(args.phase_batch_size,
                               pcfg.get("phase_batch_size", 0) or 0,
                               "phase_batch_size") or 0)
    converter_extra_args = pick(args.converter_extra_args,
                                pcfg.get("converter_extra_args", "") or "",
                                "converter_extra_args") or ""
    render_video = bool(pcfg.get("render_video", True))
    video_config = pcfg.get("video_config", "") or ""
    reannotate = bool(pcfg.get("reannotate", False))
    reannotate_config_arg = pcfg.get("reannotate_config", "") or ""

    exclude_subdirs_raw = str(pcfg.get("exclude_subdirs", "calibration") or "")
    if exclude_subdirs_raw.strip().lower() in ("", "none"):
        exclude_subdirs = set()
    else:
        exclude_subdirs = {s.strip() for s in exclude_subdirs_raw.split(",") if s.strip()}

    scratch_root = args.data_dir or tempfile.gettempdir()
    stop_memory_logger = start_memory_logger(scratch_root)

    cpu_budget = available_cpus()
    max_total = int(pick(args.max_total_converter_workers,
                         pcfg.get("max_total_converter_workers", 0) or 0,
                         "max_total_converter_workers") or 0)
    worker_budget = min(cpu_budget, max_total) if max_total > 0 else cpu_budget
    converter_workers = max(1, worker_budget // max(1, num_conversion_workers))
    if "--num_workers" in converter_extra_args:
        print(f"[cpu] budget {cpu_budget} cpus; --num_workers pinned by "
              f"converter_extra_args ({converter_extra_args.strip()})")
    else:
        print(f"[cpu] budget {cpu_budget} cpus, worker budget {worker_budget} / "
              f"{num_conversion_workers} concurrent bag(s) -> --num_workers "
              f"{converter_workers} each ({converter_workers * num_conversion_workers} total)")

    scan_location = f"{stager.remote}:{args.root_dir}" if stager else os.path.join(args.data_dir, args.root_dir)
    print(f"[discovery] scanning {scan_location} (exclude_subdirs={sorted(exclude_subdirs)})")
    run_dirs = find_rosbag_run_dirs(args.root_dir, exclude_subdirs, limit_subfolder=args.limit_subfolder,
                                     stager=stager, data_dir=args.data_dir)
    print(f"[discovery] found {len(run_dirs)} rosbag run-dir(s)")

    pending = filter_unconverted(run_dirs, args.dst_dir, stager=stager, data_dir=args.data_dir)
    print(f"[discovery] {len(pending)} pending (not yet converted)")

    # reannotated_dir/record_drain come from reannotate_config's `reannotation:` block.
    reannotate_config, reannotated_dir = "", ""
    record_drain = 3.0
    if reannotate:
        assert reannotate_config_arg, "pipeline.reannotate_config is required when pipeline.reannotate is true"
        reannotate_config = resolve_deploy_path(reannotate_config_arg)
        with open(reannotate_config) as f:
            rsec = (yaml.safe_load(f) or {}).get("reannotation", {}) or {}
        reannotated_dir = rsec.get("reannotated_dir", "") or ""
        record_drain = float(rsec.get("record_drain", 3.0) or 3.0)
        print(f"[reannotate] enabled (config={reannotate_config}) -- bags missing "
              f"super_odometry will be reannotated first, {num_reannotation_workers} "
              f"at a time")
        if reannotated_dir:
            print(f"[reannotate] merged reannotated bags will also be uploaded to "
                  f"{reannotated_dir}")

    orig_scratch_root = tempfile.mkdtemp(prefix="osmo_pipeline_orig_")
    merged_scratch_root = tempfile.mkdtemp(prefix="osmo_pipeline_merged_")
    kitti_out_root = args.local_out_dir or tempfile.mkdtemp(prefix="osmo_pipeline_kittiscratch_")

    summary_path = os.path.join(args.local_out_dir, "_pipeline_summary.json") if args.local_out_dir else None
    summary = []

    def flush_summary():
        if summary_path:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

    domain_queue = None
    manager = None
    if reannotate and num_reannotation_workers > 0:
        manager = multiprocessing.Manager()
        domain_queue = manager.Queue()
        for k in range(num_reannotation_workers):
            domain_queue.put(1 + (k % 100))  # ROS_DOMAIN_ID kept within 1..100

    succeeded, failed = [], []
    batches = list(chunked(pending, phase_batch_size))

    for batch_num, batch in enumerate(batches, start=1):
        if not batch:
            continue
        if len(batches) > 1:
            print(f"\n[batch] {batch_num}/{len(batches)}: {len(batch)} bag(s)")

        merged_by_relpath = {}
        if reannotate:
            needs_reannotation = classify_reannotation_needed(batch, args.root_dir, stager, args.data_dir)
            print(f"[reannotate] {len(needs_reannotation)}/{len(batch)} bag(s) in this "
                  f"batch need reannotation")
            if needs_reannotation:
                submit_times = {}
                with ProcessPoolExecutor(max_workers=num_reannotation_workers,
                                         initializer=_init_worker_stdio) as pool:
                    futures = {}
                    for relpath in needs_reannotation:
                        submit_times[relpath] = time.time()
                        futures[pool.submit(
                            reannotate_one, relpath, args.root_dir, kitti_out_root,
                            orig_scratch_root, merged_scratch_root, args.dst_dir,
                            args.pipeline_config, reannotate_config, reannotated_dir,
                            record_drain, domain_queue, stager, args.data_dir,
                        )] = relpath
                    for future in as_completed(futures):
                        relpath, ok, err, merged_dir = future.result()
                        duration_s = time.time() - submit_times[relpath]
                        if ok:
                            print(f"[reannotate] OK   {relpath}", flush=True)
                            if merged_dir:
                                merged_by_relpath[relpath] = merged_dir
                        else:
                            print(f"[reannotate] FAIL {relpath}: {err}", flush=True)
                            failed.append(relpath)
                        summary.append({
                            "relpath": relpath, "phase": "reannotate",
                            "status": "ok" if ok else "fail", "reason": err,
                            "duration_s": round(duration_s, 1),
                        })
                        flush_summary()

        # Bags whose reannotation failed don't go on to conversion.
        to_convert = [relpath for relpath in batch if relpath not in failed]

        if to_convert:
            submit_times = {}
            with ProcessPoolExecutor(max_workers=num_conversion_workers,
                                     initializer=_init_worker_stdio) as pool:
                futures = {}
                for relpath in to_convert:
                    merged_dir = merged_by_relpath.get(relpath)
                    if merged_dir is None:
                        orig_dir = None
                    elif stager:
                        orig_dir = os.path.join(orig_scratch_root, relpath)
                    else:
                        orig_dir = os.path.join(args.data_dir, args.root_dir, relpath)
                    merged_scratch_dir = os.path.join(merged_scratch_root, relpath) if merged_dir else None
                    submit_times[relpath] = time.time()
                    futures[pool.submit(
                        convert_one, relpath, args.root_dir, args.dst_dir,
                        args.pipeline_config, converter_extra_args,
                        render_video, video_config, kitti_out_root,
                        stager, args.data_dir, args.local_out_dir, converter_workers,
                        merged_dir, orig_dir, merged_scratch_dir,
                    )] = relpath
                for future in as_completed(futures):
                    relpath, ok, err, kept_frames = future.result()
                    duration_s = time.time() - submit_times[relpath]
                    if ok:
                        print(f"[convert] OK   {relpath}", flush=True)
                        succeeded.append(relpath)
                    else:
                        print(f"[convert] FAIL {relpath}: {err}", flush=True)
                        failed.append(relpath)
                    summary.append({
                        "relpath": relpath, "phase": "convert",
                        "status": "ok" if ok else "fail", "reason": err,
                        "duration_s": round(duration_s, 1), "kept_frames": kept_frames,
                    })
                    flush_summary()

    if manager is not None:
        manager.shutdown()

    shutil.rmtree(orig_scratch_root, ignore_errors=True)
    shutil.rmtree(merged_scratch_root, ignore_errors=True)
    if not args.local_out_dir:
        shutil.rmtree(kitti_out_root, ignore_errors=True)

    stop_memory_logger()  # quiet the [mem]/[disk] heartbeat now that all work is done

    print(
        f"[summary] found={len(run_dirs)} pending={len(pending)} "
        f"converted={len(succeeded)} failed={len(failed)}"
    )
    if failed:
        print("[summary] failed run-dirs:")
        for relpath in failed:
            print(f"    {relpath}")
        sys.exit(1)


if __name__ == "__main__":
    main()
