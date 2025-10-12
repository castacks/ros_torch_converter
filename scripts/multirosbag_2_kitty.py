#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively find folders that contain .mcap files and run a converter once per folder, preserving structure under an output root."
    )
    parser.add_argument(
        "-r", "--root", required=True, type=Path,
        help="Dataset root to scan recursively (e.g., /airlab_storage/tartanRGBT/dataset)",
    )
    parser.add_argument(
        "-o", "--out-root", type=Path, default=None,
        help="Output root (default: <root>/extracted_data)",
    )
    parser.add_argument(
        "-c", "--config", type=Path,
        default=Path("src/core/ros_torch_converter/config/kitti_config/tartan_rgbt.yaml"),
        help="Path to converter config YAML",
    )
    parser.add_argument(
        "-C", "--converter", type=Path,
        default=Path("src/core/ros_torch_converter/scripts/ros2bag_2_kitti.py"),
        help="Path to ros2bag_2_kitti.py (or compatible) converter",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "-s", "--skip-if-exists", action="store_true",
        help="Skip a destination if it already contains files",
    )
    parser.add_argument(
        "--ext", default=".mcap",
        help="File extension to search for (default: .mcap)",
    )
    return parser.parse_args()


def find_leaf_dirs_with_ext(root: Path, ext: str) -> List[Path]:
    """
    Return unique directories under `root` that contain at least one file with extension `ext`.
    Assumes: a folder containing any `ext` file will not have subfolders that also contain `ext`.
    """
    # Use rglob to find all files, then deduplicate parents
    parents: Set[Path] = set()
    # Normalize ext to start with dot
    if not ext.startswith("."):
        ext = "." + ext
    for p in root.rglob(f"*{ext}"):
        if p.is_file():
            parents.add(p.parent)
    # Sort for stable processing order
    return sorted(parents)


def dst_has_files(dst: Path) -> bool:
    if not dst.exists():
        return False
    try:
        next(dst.iterdir())
        return True
    except StopIteration:
        return False


def main():
    args = parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: root not found or not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    out_root = (args.out_root or (root / "extracted_data")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    converter = args.converter.resolve()
    config = args.config.resolve()

    print(f"Dataset root   : {root}")
    print(f"Output root    : {out_root}")
    print(f"Converter      : {converter}")
    print(f"Config         : {config}")
    print(f"Dry run        : {args.dry_run}")
    print(f"Skip if exists : {args.skip_if_exists}")
    print(f"Search ext     : {args.ext}")
    print()

    # It’s okay if converter/config don’t exist yet (e.g., inside venv/cwd),
    # but warn early to prevent silent failures.
    if not converter.exists():
        print(f"Warning: converter not found at {converter}", file=sys.stderr)
    if not config.exists():
        print(f"Warning: config not found at {config}", file=sys.stderr)

    bag_dirs = find_leaf_dirs_with_ext(root, args.ext)
    if not bag_dirs:
        print(f"No *{args.ext} files found under {root}")
        sys.exit(0)

    for src_dir in bag_dirs:
        try:
            rel = src_dir.relative_to(root)
        except ValueError:
            # Fallback if for some reason src_dir isn't under root after resolves
            rel = Path(os.path.relpath(src_dir, root))

        dst_dir = (out_root / rel)

        if args.skip_if_exists and dst_has_files(dst_dir):
            print(f"[SKIP] {src_dir} -> {dst_dir} (already contains files)")
            continue
        elif dst_dir.exists():
            dst_dir.rmdir()

        

        print(f"[RUN ] {src_dir} -> {dst_dir}")

        if args.dry_run:
            print(
                f"       python3 \"{converter}\" --config \"{config}\" "
                f"--src_dir \"{src_dir}\" --dst_dir \"{dst_dir}\""
            )
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Run converter once per folder
        cmd = [
            sys.executable or "python3",
            "-u",  # <-- unbuffered stdout/stderr
            str(converter),
            "--config", str(config),
            "--src_dir", str(src_dir),
            "--dst_dir", str(dst_dir),
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERR ] Converter failed for {src_dir}: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
