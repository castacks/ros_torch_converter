#!/usr/bin/env python3
"""Rebuild a KITTI dataset tree from the packed-HDF5 mirror.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))
from ros_torch_converter.kitti_hdf5 import (  # noqa: E402
    read_bytes_dataset, read_manifest, restore_per_frame_group, _name_key)


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def restore_meta(meta_h5, dst_dir):
    """Restore every root file + the whole tf/ tree from _meta.h5.

    Returns (n_root_files, n_tf_files).
    """
    os.makedirs(dst_dir, exist_ok=True)
    with h5py.File(meta_h5, "r") as f:
        root_names = json.loads(f["root"].attrs["names"])
        for name in root_names:
            _write(os.path.join(dst_dir, name),
                   read_bytes_dataset(f["root"], _name_key(name)))
        tf_rel = json.loads(f["tf"].attrs["relpaths"]) if "tf" in f else []
        for rel in tf_rel:
            dst = os.path.join(dst_dir, "tf", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _write(dst, read_bytes_dataset(f["tf"], rel))
    return len(root_names), len(tf_rel)


def restore_modality(h5_path, dst_dir, frame_slice=None):
    """Restore one modality .h5 into ``dst_dir/<group>/<name>/``. Returns files written."""
    with h5py.File(h5_path, "r") as f:
        group = f.attrs["group"]
        name = f.attrs["name"]
        mod_dir = os.path.join(dst_dir, group, name)
        os.makedirs(mod_dir, exist_ok=True)
        n = 0
        for fname in json.loads(f["files"].attrs["names"]):
            _write(os.path.join(mod_dir, fname),
                   read_bytes_dataset(f["files"], _name_key(fname)))
            n += 1
        if "frames" in f:
            for key in f["frames"]:
                n += restore_per_frame_group(f["frames"][key], mod_dir,
                                             frame_slice=frame_slice)
    return f"{group}/{name}", n


def _restore_worker(args):
    h5_path, dst_dir, frame_slice = args
    try:
        modname, n = restore_modality(h5_path, dst_dir, frame_slice)
        return modname, n, None
    except Exception:
        import traceback
        return os.path.basename(h5_path), 0, traceback.format_exc()


def pull_remote(remote, remote_dir, local_dir, wanted_relpaths):
    """rclone-copy _meta.h5 + the wanted modality .h5 files into local_dir."""
    from tartandriver_utils.rclone_stager import RcloneStager
    stager = RcloneStager(remote=remote)
    os.makedirs(local_dir, exist_ok=True)
    for rel in ["_meta.h5"] + list(wanted_relpaths):
        dst_sub = os.path.join(local_dir, os.path.dirname(rel))
        os.makedirs(dst_sub, exist_ok=True)
        print(f"[unpack] rclone {remote}:{remote_dir}/{rel} -> {dst_sub}")
        stager.copy_in(os.path.join(remote_dir, rel), dst_sub)


def print_manifest(manifest, src_dir):
    print(f"{'modality':40s} {'kind':10s} {'frames':>8s} {'packed':>10s} "
          f"{'src_files':>10s}  local?")
    for e in manifest:
        local = os.path.exists(os.path.join(src_dir, e["h5_relpath"]))
        packed = "-"
        if local:
            packed = f"{os.path.getsize(os.path.join(src_dir, e['h5_relpath']))/1e6:.1f}MB"
        print(f"{e['group']+'/'+e['name']:40s} {e['kind']:10s} "
              f"{e['n_frames']:8d} {packed:>10s} {e['src_file_count']:10d}  "
              f"{'yes' if local else 'NO'}")


def parse_frames(spec):
    if not spec:
        return None
    a, _, b = spec.partition(":")
    return (int(a or 0), int(b) if b else 1 << 62)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", required=True, help="dir holding the .h5 mirror")
    ap.add_argument("--dst_dir", default="", help="output KITTI tree (not needed for --list)")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    ap.add_argument("--modalities", default="", help="comma-separated group/name allowlist")
    ap.add_argument("--frames", default="", help="restore only this frame range, e.g. 0:100")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--remote", default="", help="rclone remote to pull the .h5 files from first")
    ap.add_argument("--remote_dir", default="", help="path of the run on --remote")
    args = ap.parse_args(argv)

    src_dir = os.path.abspath(args.src_dir)
    allow = {m.strip() for m in args.modalities.split(",") if m.strip()}

    if args.remote:
        if not args.remote_dir:
            ap.error("--remote requires --remote_dir")
        # need the manifest to know which modality files to pull; grab _meta first
        from tartandriver_utils.rclone_stager import RcloneStager
        os.makedirs(src_dir, exist_ok=True)
        RcloneStager(remote=args.remote).copy_in(
            os.path.join(args.remote_dir, "_meta.h5"), src_dir)
        with h5py.File(os.path.join(src_dir, "_meta.h5"), "r") as f:
            man = read_manifest(f)
        wanted = [e["h5_relpath"] for e in man
                  if not allow or f"{e['group']}/{e['name']}" in allow]
        pull_remote(args.remote, args.remote_dir, src_dir, wanted)

    meta_h5 = os.path.join(src_dir, "_meta.h5")
    if not os.path.exists(meta_h5):
        sys.exit(f"[unpack] {meta_h5} not found -- it is required. Fetch it with:\n"
                 f"    rclone copy <remote>:<run_dir>/_meta.h5 {src_dir}")
    with h5py.File(meta_h5, "r") as f:
        manifest = read_manifest(f)
        fmt = f.attrs.get("format_version", "?")

    if args.list:
        print(f"[unpack] {src_dir}  (format v{fmt}, {len(manifest)} modalities)")
        print_manifest(manifest, src_dir)
        return 0

    if not args.dst_dir:
        ap.error("--dst_dir is required unless --list")
    dst_dir = os.path.abspath(args.dst_dir)
    frame_slice = parse_frames(args.frames)

    t0 = time.time()
    n_root, n_tf = restore_meta(meta_h5, dst_dir)
    print(f"[unpack] restored {n_root} root file(s) + {n_tf} tf/ file(s) from _meta.h5")

    present, missing = [], []
    for e in manifest:
        if allow and f"{e['group']}/{e['name']}" not in allow:
            continue
        p = os.path.join(src_dir, e["h5_relpath"])
        (present if os.path.exists(p) else missing).append(e)

    # also restore any .h5 on disk that predates / is absent from the manifest
    known = {e["h5_relpath"] for e in manifest}
    for p in glob.glob(os.path.join(src_dir, "**", "*.h5"), recursive=True):
        rel = os.path.relpath(p, src_dir)
        if rel == "_meta.h5" or rel in known:
            continue
        present.append({"group": "?", "name": rel, "h5_relpath": rel})

    workers = args.workers or min(8, max(1, len(present)))
    tasks = [(os.path.join(src_dir, e["h5_relpath"]), dst_dir, frame_slice)
             for e in present]
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks) or 1))) as pool:
        for fut in as_completed({pool.submit(_restore_worker, t): t for t in tasks}):
            modname, n, err = fut.result()
            if err:
                failed.append(modname)
                print(f"[unpack] FAIL {modname}\n{err}", flush=True)
            else:
                done.append(modname)
                print(f"[unpack] {modname}: {n} file(s)", flush=True)

    viz_src = os.path.join(src_dir, "viz")
    if os.path.isdir(viz_src) and not frame_slice:
        shutil.copytree(viz_src, os.path.join(dst_dir, "viz"), dirs_exist_ok=True)

    print(f"[unpack] done in {time.time()-t0:.0f}s: restored {len(done)}/"
          f"{len(manifest)} modalities into {dst_dir}")
    if missing:
        names = ", ".join(f"{e['group']}/{e['name']}" for e in missing)
        print(f"[unpack] {len(missing)} modality/-ies in the manifest were not "
              f"downloaded: {names}")
    if frame_slice:
        print(f"[unpack] NOTE: --frames {args.frames} was set -- this is a "
              f"partial tree, not a complete KITTI dataset")
    if failed:
        print(f"[unpack] {len(failed)} modality/-ies FAILED to restore")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
