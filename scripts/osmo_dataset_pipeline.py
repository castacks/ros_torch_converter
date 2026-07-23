"""OSMO dataset discovery + conversion pipeline driver."""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
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


def convert_one(relpath, root_dir, dst_dir, pipeline_config, converter_extra_args,
                render_video=True, video_config="", reannotate=False, reannotate_config="",
                reannotated_dir="", reannotate_timeout=None, domain_queue=None,
                stager=None, data_dir=None, local_out_dir="", converter_workers=0):
    scratch_dir = tempfile.mkdtemp(prefix="osmo_pipeline_raw_")
    # When local_out_dir is set the converted dataset is written to a known,
    # persistent location (<local_out_dir>/<relpath>) that the caller can read
    # afterwards; otherwise it goes to a scratch tempdir deleted after copy-back.
    if local_out_dir:
        out_dir = os.path.join(local_out_dir, relpath)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = tempfile.mkdtemp(prefix="osmo_pipeline_kitti_")
    reann_dir = None
    domain_id = None

    try:
        if stager:
            print(f"[convert] {relpath}: copying raw bag from {stager.remote} ...")
            stager.copy_in(os.path.join(root_dir, relpath), scratch_dir)
            src_dir = scratch_dir
        else:
            src_dir = os.path.join(data_dir, root_dir, relpath)

        # Reannotate with super_odometry first if the raw bag is missing it.
        if reannotate and not bag_has_superodometry(src_dir):
            orig_src = src_dir
            # Checked out for this bag's playback only, then released below.
            if domain_queue is not None:
                domain_id = domain_queue.get()
            print(f"[reannotate] {relpath}: missing super_odometry -- reannotating "
                  f"(ROS_DOMAIN_ID={domain_id}) ...")
            reann_dir = tempfile.mkdtemp(prefix="osmo_pipeline_reann_")
            reann_kwargs = {"config_path": reannotate_config} if reannotate_config else {}
            src_dir = reannotate_bag(
                orig_src, reann_dir, domain_id=domain_id,
                timeout=reannotate_timeout, **reann_kwargs,
            )

            print(f"[reannotate] {relpath}: writing message-count report + sync viz ...")
            report_rows = build_report(orig_src, src_dir)
            write_report(report_rows, os.path.join(out_dir, "reannotate_report"))
            try:
                render_sync_viz(src_dir, resolve_config_path(pipeline_config),
                                 os.path.join(out_dir, "topic_sync_viz.png"))
            except Exception as e:
                print(f"[reannotate] {relpath}: topic_sync_viz failed (non-fatal): {e}")

            if reannotated_dir:
                # reannotated_dir is relative to dst_dir (same root as the KITTI copy-back).
                bag_dst_relpath = os.path.join(dst_dir, reannotated_dir, relpath)
                if stager:
                    print(f"[reannotate] {relpath}: copying reannotated bag to "
                          f"{stager.remote}:{bag_dst_relpath} ...")
                    stager.copy_out(src_dir, bag_dst_relpath)
                else:
                    print(f"[reannotate] {relpath}: no stager configured -- skipping "
                          f"reannotated_dir save")

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

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return relpath, False, f"conversion exited with code {proc.returncode}"

        dst_relpath = os.path.join(dst_dir, relpath)
        if stager:
            print(f"[convert] {relpath}: copying result to {stager.remote}:{dst_relpath} ...")
            stager.copy_out(out_dir, dst_relpath)
        else:
            local_dst = os.path.join(data_dir, dst_relpath)
            print(f"[convert] {relpath}: copying result to {local_dst} ...")
            shutil.copytree(out_dir, local_dst, dirs_exist_ok=True)
        return relpath, True, None
    except subprocess.CalledProcessError as e:
        return relpath, False, str(e)
    finally:
        if domain_id is not None and domain_queue is not None:
            domain_queue.put(domain_id)  # release the domain for the next bag
        shutil.rmtree(scratch_dir, ignore_errors=True)
        if not local_out_dir:  # keep the dataset when a known output dir was requested
            shutil.rmtree(out_dir, ignore_errors=True)
        if reann_dir is not None:
            shutil.rmtree(reann_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="root to scan for run-dirs (rclone-remote-relative, or data_dir-relative if --data_dir is set)")
    parser.add_argument("--dst_dir", required=True, help="destination for the final KITTI copy-back (rclone-remote-relative, or data_dir-relative if --data_dir is set)")
    parser.add_argument("--data_dir", default="", help="local mount root for full-copy mode (bulk_copy.sh); when set, reads/writes go straight to this mount instead of through an RcloneStager")
    parser.add_argument("--pipeline_config", required=True,
                        help="path to a ros_torch_converter config yaml")
    parser.add_argument("--limit_subfolder", default=None, help="restrict discovery to a single run-dir relpath")
    parser.add_argument("--local_out_dir", default="", help="if set, write each converted KITTI dataset to <local_out_dir>/<relpath> and keep it there (instead of a scratch tempdir deleted after copy-back), so the caller can read results locally")
    return parser.parse_args()


def main():
    args = parse_args()

    stager = RcloneStager() if not args.data_dir else None

    cfg = load_yaml(resolve_config_path(args.pipeline_config))
    pcfg = cfg.get("pipeline", {}) or {}

    num_conversion_workers = int(pcfg.get("num_conversion_workers", 4))
    converter_extra_args = pcfg.get("converter_extra_args", "") or ""
    render_video = bool(pcfg.get("render_video", True))
    video_config = pcfg.get("video_config", "") or ""
    reannotate = bool(pcfg.get("reannotate", False))
    reannotate_config_arg = pcfg.get("reannotate_config", "") or ""

    exclude_subdirs_raw = str(pcfg.get("exclude_subdirs", "calibration") or "")
    if exclude_subdirs_raw.strip().lower() in ("", "none"):
        exclude_subdirs = set()
    else:
        exclude_subdirs = {s.strip() for s in exclude_subdirs_raw.split(",") if s.strip()}

    # Share the pod's CPU budget across the bags in flight rather than letting each
    # converter claim all of it (see convert_one).
    cpu_budget = available_cpus()
    converter_workers = max(1, cpu_budget // max(1, num_conversion_workers))
    print(f"[cpu] budget {cpu_budget} cpus / {num_conversion_workers} concurrent bag(s) "
          f"-> --num_workers {converter_workers} per conversion")

    scan_location = f"{stager.remote}:{args.root_dir}" if stager else os.path.join(args.data_dir, args.root_dir)
    print(f"[discovery] scanning {scan_location} (exclude_subdirs={sorted(exclude_subdirs)})")
    run_dirs = find_rosbag_run_dirs(args.root_dir, exclude_subdirs, limit_subfolder=args.limit_subfolder,
                                     stager=stager, data_dir=args.data_dir)
    print(f"[discovery] found {len(run_dirs)} rosbag run-dir(s)")

    pending = filter_unconverted(run_dirs, args.dst_dir, stager=stager, data_dir=args.data_dir)
    print(f"[discovery] {len(pending)} pending (not yet converted)")

    # Stack + reannotated_dir/timeout all come from reannotate_config's `reannotation:` block.
    reannotate_config = ""
    reannotated_dir, reannotate_timeout = "", None
    domain_queue = None
    manager = None
    if reannotate:
        assert reannotate_config_arg, "pipeline.reannotate_config is required when pipeline.reannotate is true"
        reannotate_config = resolve_deploy_path(reannotate_config_arg)
        with open(reannotate_config) as f:
            rsec = (yaml.safe_load(f) or {}).get("reannotation", {}) or {}
        reannotated_dir = rsec.get("reannotated_dir", "") or ""
        t = str(rsec.get("timeout", "") or "").strip().lower()
        reannotate_timeout = float(t) if t not in ("", "none") else None
        print(f"[reannotate] enabled (config={reannotate_config}) -- bags missing "
              f"super_odometry will be reannotated first")
        if reannotated_dir:
            print(f"[reannotate] merged reannotated bags will also be uploaded to "
                  f"{reannotated_dir}")

        # One distinct ROS_DOMAIN_ID per worker guarantees no two concurrent bags collide.
        manager = multiprocessing.Manager()
        domain_queue = manager.Queue()
        for k in range(num_conversion_workers):
            domain_queue.put(1 + (k % 100))  # ROS_DOMAIN_ID kept within 1..100

    succeeded, failed = [], []
    if pending:
        with ProcessPoolExecutor(max_workers=num_conversion_workers) as pool:
            futures = {
                pool.submit(
                    convert_one, relpath, args.root_dir, args.dst_dir,
                    args.pipeline_config, converter_extra_args,
                    render_video, video_config,
                    reannotate, reannotate_config, reannotated_dir,
                    reannotate_timeout, domain_queue,
                    stager, args.data_dir, args.local_out_dir, converter_workers,
                ): relpath
                for relpath in pending
            }
            for future in as_completed(futures):
                relpath, ok, err = future.result()
                if ok:
                    print(f"[convert] OK   {relpath}")
                    succeeded.append(relpath)
                else:
                    print(f"[convert] FAIL {relpath}: {err}")
                    failed.append(relpath)

    if manager is not None:
        manager.shutdown()

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
