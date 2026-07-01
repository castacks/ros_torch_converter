"""
OSMO dataset discovery + conversion pipeline driver.
"""

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERTER_SCRIPT = os.path.join(SCRIPT_DIR, "ros2bag_2_kitti_multiproc.py")
CONVERTER_PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)  # .../ros_torch_converter/


def resolve_kitti_config(kitti_config):
    if os.path.isabs(kitti_config):
        return kitti_config
    return os.path.join(CONVERTER_PACKAGE_DIR, kitti_config)


def rclone_lsjson_recursive(remote_path):
    """List an rclone remote path recursively; returns rclone's parsed JSON entries."""
    result = subprocess.run(
        ["rclone", "lsjson", "--recursive", remote_path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def find_rosbag_run_dirs(root_dir, exclude_subdirs, limit_subfolder=None):
    """
    Find run-dirs under `airlab_storage:<root_dir>` that look like a valid
    rosbag dir (has metadata.yaml + >=1 *.mcap), at any depth. `root_dir` can
    be a top-level root containing many date folders (<date>/<subfolder>/<run_dir>)
    or a single date folder directly (<subfolder>/<run_dir>) -- no fixed
    depth is assumed. A run-dir is skipped if any component of its relpath
    (e.g. "calibration") is in `exclude_subdirs`, regardless of position.
    """
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

        parts = relpath.split("/")
        if any(part in exclude_subdirs for part in parts):
            continue

        has_metadata = "metadata.yaml" in filenames
        has_mcaps = any(f.endswith(".mcap") for f in filenames)
        if has_metadata and has_mcaps:
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
        subprocess.run(
            ["rclone", "copy", f"{RCLONE_REMOTE}:{os.path.join(root_dir, relpath)}", scratch_dir],
            check=True,
        )

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

        subprocess.run(
            ["rclone", "copy", out_dir, f"{RCLONE_REMOTE}:{os.path.join(dst_dir, relpath)}"],
            check=True,
        )
        return relpath, True, None
    except subprocess.CalledProcessError as e:
        return relpath, False, str(e)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="rclone airlab_storage: path to scan -- either the top-level root (containing many date folders) or a single date folder directly")
    parser.add_argument("--dst_dir", required=True, help="rclone airlab_storage: path for the final KITTI copy-back")
    parser.add_argument("--kitti_input_mount", required=True, help="local mount of the OSMO S3 kitti-output input ({{input:0}})")
    parser.add_argument("--kitti_output_mount", required=True, help="local mount of the OSMO S3 kitti-output output ({{output}})")
    parser.add_argument("--exclude_subdirs", default="calibration", help="comma-separated subfolder names to skip anywhere in the path (e.g. calibration), or 'none'/'' to exclude nothing")
    parser.add_argument("--kitti_config", required=True, help="path to the ros_torch_converter kitti config yaml")
    parser.add_argument("--num_conversion_workers", type=int, default=4)
    parser.add_argument("--converter_extra_args", default="", help="passthrough args for ros2bag_2_kitti_multiproc.py, e.g. '--render_video_hud'")
    parser.add_argument(
        "--limit_subfolder", default=None,
        help="TEMPORARY: restrict discovery to this single run-dir relpath, for validating the pipeline end-to-end. Delete this flag once the pipeline is trusted.",
    )
    args = parser.parse_args()

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
