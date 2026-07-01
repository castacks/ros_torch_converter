"""OSMO dataset discovery + conversion pipeline driver."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from tartandriver_utils.os_utils import is_kitti_dir


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


def convert_one(relpath, root_dir, dst_dir, kitti_output_mount, kitti_config, converter_extra_args):
    scratch_dir = tempfile.mkdtemp(prefix="osmo_pipeline_raw_")
    out_dir = os.path.join(kitti_output_mount, relpath)
    os.makedirs(out_dir, exist_ok=True)

    try:
        print(f"[convert] {relpath}: copying raw bag from {RCLONE_REMOTE} ...")
        rclone_copy(f"{RCLONE_REMOTE}:{os.path.join(root_dir, relpath)}", scratch_dir)

        cmd = [
            sys.executable, CONVERTER_SCRIPT,
            "--config", resolve_kitti_config(kitti_config),
            "--src_dir", scratch_dir,
            "--dst_dir", out_dir,
            "--force",
        ]
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
        shutil.rmtree(scratch_dir, ignore_errors=True)


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
    return parser.parse_args()


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

    succeeded, failed = [], []
    if pending:
        with ProcessPoolExecutor(max_workers=args.num_conversion_workers) as pool:
            futures = {
                pool.submit(
                    convert_one, relpath, args.root_dir, args.dst_dir,
                    args.kitti_output_mount, args.kitti_config, args.converter_extra_args,
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
