"""OSMO dataset discovery + conversion pipeline driver."""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from tartandriver_utils.os_utils import is_kitti_dir

from osmo_reannotation import bag_has_superodometry, reannotate_bag, resolve_deploy_path


RCLONE_REMOTE = "airlab_storage"

# --multi-thread-streams=0: airlab_storage (FTP) mishandles chunked downloads.
# --stats-one-line: avoid carriage-return redraws in captured task logs.
RCLONE_FLAGS = ["--multi-thread-streams=0", "--transfers=10", "--stats=15s", "--stats-one-line"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERTER_SCRIPT = os.path.join(SCRIPT_DIR, "ros2bag_2_kitti_multiproc.py")
CONVERTER_PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)


def resolve_kitti_config(kitti_config):
    if os.path.isabs(kitti_config):
        return kitti_config
    return os.path.join(CONVERTER_PACKAGE_DIR, kitti_config)


def rclone_copy(src, dst):
    subprocess.run(["rclone", "copy", src, dst] + RCLONE_FLAGS, check=True)


def rclone_lsjson_recursive(remote_path):
    result = subprocess.run(
        ["rclone", "lsjson", "--recursive", remote_path],
        stdout=subprocess.PIPE, text=True, check=True,
    )
    return json.loads(result.stdout)


def find_rosbag_run_dirs(root_dir, exclude_subdirs, limit_subfolder=None):
    """Find run-dirs under airlab_storage:root_dir with metadata.yaml + >=1 .mcap, at any depth."""
    entries = rclone_lsjson_recursive(f"{RCLONE_REMOTE}:{root_dir}")

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
        if "metadata.yaml" in filenames and any(f.endswith(".mcap") for f in filenames):
            run_dirs.append(relpath)

    return sorted(run_dirs)


def filter_unconverted(run_dirs, kitti_input_mount):
    """Drop run-dirs that already have a completed KITTI dataset at the destination."""
    return [
        relpath for relpath in run_dirs
        if not is_kitti_dir(os.path.join(kitti_input_mount, relpath))
    ]


def convert_one(relpath, root_dir, dst_dir, kitti_output_mount, kitti_config,
                converter_extra_args, render_video=True, video_config="",
                reannotate=False, reannotate_config="", reannotated_dir="",
                reannotate_timeout=None, domain_queue=None):
    scratch_dir = tempfile.mkdtemp(prefix="osmo_pipeline_raw_")
    reann_dir = None
    domain_id = None
    out_dir = os.path.join(kitti_output_mount, relpath)
    os.makedirs(out_dir, exist_ok=True)

    try:
        print(f"[convert] {relpath}: copying raw bag from {RCLONE_REMOTE} ...")
        rclone_copy(f"{RCLONE_REMOTE}:{os.path.join(root_dir, relpath)}", scratch_dir)

        # Reannotate with super_odometry first if the raw bag is missing it.
        src_dir = scratch_dir
        if reannotate and not bag_has_superodometry(scratch_dir):
            # Checked out for this bag's playback only, then released below.
            if domain_queue is not None:
                domain_id = domain_queue.get()
            print(f"[reannotate] {relpath}: missing super_odometry -- reannotating "
                  f"(ROS_DOMAIN_ID={domain_id}) ...")
            reann_dir = tempfile.mkdtemp(prefix="osmo_pipeline_reann_")
            reann_kwargs = {"config_path": reannotate_config} if reannotate_config else {}
            src_dir = reannotate_bag(
                scratch_dir, reann_dir, domain_id=domain_id,
                timeout=reannotate_timeout, **reann_kwargs,
            )
            if reannotated_dir:
                print(f"[reannotate] {relpath}: copying reannotated bag to "
                      f"{RCLONE_REMOTE}:{reannotated_dir} ...")
                rclone_copy(src_dir, f"{RCLONE_REMOTE}:{os.path.join(reannotated_dir, relpath)}")

        cmd = [
            sys.executable, CONVERTER_SCRIPT,
            "--config", resolve_kitti_config(kitti_config),
            "--src_dir", src_dir,
            "--dst_dir", out_dir,
            "--force",
        ]
        if not render_video:
            cmd.append("--no_render_video")
        elif video_config:
            cmd.extend(["--video_config", resolve_kitti_config(video_config)])
        if converter_extra_args:
            cmd.extend(converter_extra_args.split())

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return relpath, False, f"conversion exited with code {proc.returncode}"

        print(f"[convert] {relpath}: copying result to {RCLONE_REMOTE}:{dst_dir} ...")
        rclone_copy(out_dir, f"{RCLONE_REMOTE}:{os.path.join(dst_dir, relpath)}")
        return relpath, True, None
    except subprocess.CalledProcessError as e:
        return relpath, False, str(e)
    finally:
        if domain_id is not None and domain_queue is not None:
            domain_queue.put(domain_id)  # release the domain for the next bag
        shutil.rmtree(scratch_dir, ignore_errors=True)
        if reann_dir is not None:
            shutil.rmtree(reann_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="rclone airlab_storage: root to scan for run-dirs")
    parser.add_argument("--dst_dir", required=True, help="rclone airlab_storage: path for the final KITTI copy-back")
    parser.add_argument("--kitti_input_mount", required=True, help="local mount to check for already-converted datasets")
    parser.add_argument("--kitti_output_mount", required=True, help="local mount to write converted datasets to")
    parser.add_argument("--exclude_subdirs", default="calibration", help="comma-separated subfolder names to skip, or 'none'")
    parser.add_argument("--kitti_config", required=True, help="path to the ros_torch_converter kitti config yaml")
    parser.add_argument("--num_conversion_workers", type=int, default=4)
    parser.add_argument("--converter_extra_args", default="", help="passthrough args for ros2bag_2_kitti_multiproc.py")
    parser.add_argument("--limit_subfolder", default=None, help="restrict discovery to a single run-dir relpath")
    parser.add_argument("--render_video", default="true", help="render output videos ('true'/'false')")
    parser.add_argument("--video_config", default="", help="path to a video render config yaml (HUD + PiP); relative to the converter package")
    parser.add_argument("--reannotate", default="false", help="reannotate bags missing super_odometry before converting ('true'/'false')")
    parser.add_argument("--reannotate_config", default="", help="tartandriver_deploy playback config defining the reannotation stack + knobs; required if --reannotate is true")
    return parser.parse_args()


def str2bool(s):
    return str(s).strip().lower() not in ("false", "0", "no", "off", "")


def main():
    args = parse_args()

    if args.exclude_subdirs.strip().lower() in ("", "none"):
        exclude_subdirs = set()
    else:
        exclude_subdirs = {s.strip() for s in args.exclude_subdirs.split(",") if s.strip()}

    print(f"[discovery] scanning {RCLONE_REMOTE}:{args.root_dir} (exclude_subdirs={sorted(exclude_subdirs)})")
    run_dirs = find_rosbag_run_dirs(args.root_dir, exclude_subdirs, limit_subfolder=args.limit_subfolder)
    print(f"[discovery] found {len(run_dirs)} rosbag run-dir(s)")

    pending = filter_unconverted(run_dirs, args.kitti_input_mount)
    print(f"[discovery] {len(pending)} pending (not yet converted)")

    reannotate = str2bool(args.reannotate)

    # Stack + reannotated_dir/timeout all come from reannotate_config's `reannotation:` block.
    reannotate_config = ""
    reannotated_dir, reannotate_timeout = "", None
    domain_queue = None
    manager = None
    if reannotate:
        assert args.reannotate_config, "--reannotate_config is required when --reannotate is true"
        reannotate_config = resolve_deploy_path(args.reannotate_config)
        with open(reannotate_config) as f:
            rsec = (yaml.safe_load(f) or {}).get("reannotation", {}) or {}
        reannotated_dir = rsec.get("reannotated_dir", "") or ""
        t = str(rsec.get("timeout", "") or "").strip().lower()
        reannotate_timeout = float(t) if t not in ("", "none") else None
        print(f"[reannotate] enabled (config={reannotate_config}) -- bags missing "
              f"super_odometry will be reannotated first")

        # One distinct ROS_DOMAIN_ID per worker guarantees no two concurrent bags collide.
        manager = multiprocessing.Manager()
        domain_queue = manager.Queue()
        for k in range(args.num_conversion_workers):
            domain_queue.put(1 + (k % 100))  # ROS_DOMAIN_ID kept within 1..100

    succeeded, failed = [], []
    if pending:
        with ProcessPoolExecutor(max_workers=args.num_conversion_workers) as pool:
            futures = {
                pool.submit(
                    convert_one, relpath, args.root_dir, args.dst_dir,
                    args.kitti_output_mount, args.kitti_config, args.converter_extra_args,
                    str2bool(args.render_video), args.video_config,
                    reannotate, reannotate_config, reannotated_dir,
                    reannotate_timeout, domain_queue,
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
