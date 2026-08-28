#!/usr/bin/env python3
"""Pack a converted KITTI dataset tree into the per-modality HDF5 mirror.
"""
import argparse
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
    FORMAT_VERSION, available_cpus, build_manifest_entry, scan_modality,
    write_bytes_dataset, write_manifest, write_per_frame_group, _name_key)

SKIP_TOP_LEVEL = {"tf", "viz"}


def discover_modalities(src_dir):
    """Yield ``(group, name, abspath)`` for every ``<group>/<name>/`` dir."""
    for group in sorted(os.listdir(src_dir)):
        gdir = os.path.join(src_dir, group)
        if group in SKIP_TOP_LEVEL or not os.path.isdir(gdir):
            continue
        for name in sorted(os.listdir(gdir)):
            mdir = os.path.join(gdir, name)
            if os.path.isdir(mdir):
                yield group, name, mdir


def pack_modality(mod_dir, out_h5, group, name, encode_workers=0):
    """Pack one modality dir into ``out_h5``. Returns (scan, elapsed_s).
    """
    t0 = time.time()
    scan = scan_modality(mod_dir)
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    tmp = out_h5 + ".partial"
    with h5py.File(tmp, "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["group"] = group
        f.attrs["name"] = name
        f.attrs["kind"] = scan["kind"]
        f.attrs["n_frames"] = int(scan["n_frames"])

        files_grp = f.create_group("files")
        for fname, path in scan["sidecars"].items():
            with open(path, "rb") as fh:
                write_bytes_dataset(files_grp, _name_key(fname), fh.read())
        for fname in scan["extra"]:
            with open(os.path.join(mod_dir, fname), "rb") as fh:
                write_bytes_dataset(files_grp, _name_key(fname), fh.read())
        files_grp.attrs["names"] = json.dumps(
            list(scan["sidecars"]) + list(scan["extra"]))

        if scan["per_frame"]:
            frames_grp = f.create_group("frames")
            for suffix, idx_map in sorted(scan["per_frame"].items()):
                write_per_frame_group(frames_grp, suffix, idx_map,
                                      scan["n_frames"],
                                      encode_workers=encode_workers)
    os.replace(tmp, out_h5)
    return scan, time.time() - t0


def _pack_worker(args):
    group, name, mod_dir, out_h5, encode_workers = args
    try:
        scan, dt = pack_modality(mod_dir, out_h5, group, name, encode_workers)
        return group, name, out_h5, scan, dt, None
    except Exception:  # one modality failing must not sink the run
        import traceback
        return group, name, out_h5, None, 0.0, traceback.format_exc()


def write_meta(src_dir, dst_dir, manifest):
    """_meta.h5: every root file + the whole tf/ tree (raw bytes) + manifest."""
    tmp = os.path.join(dst_dir, "_meta.h5.partial")
    with h5py.File(tmp, "w") as f:
        write_manifest(f, manifest)
        root_grp = f.create_group("root")
        root_names = []
        for entry in sorted(os.listdir(src_dir)):
            path = os.path.join(src_dir, entry)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as fh:
                write_bytes_dataset(root_grp, _name_key(entry), fh.read())
            root_names.append(entry)
        root_grp.attrs["names"] = json.dumps(root_names)

        tf_src = os.path.join(src_dir, "tf")
        if os.path.isdir(tf_src):
            tf_grp = f.create_group("tf")
            tf_rel = []
            for dirpath, _dirs, files in os.walk(tf_src):
                for fn in sorted(files):
                    rel = os.path.relpath(os.path.join(dirpath, fn), tf_src)
                    with open(os.path.join(dirpath, fn), "rb") as fh:
                        # '/' in the dataset name -> nested groups mirroring tf/
                        write_bytes_dataset(tf_grp, rel, fh.read())
                    tf_rel.append(rel)
            tf_grp.attrs["relpaths"] = json.dumps(sorted(tf_rel))
    os.replace(tmp, os.path.join(dst_dir, "_meta.h5"))


def verify(src_dir, dst_dir, manifest):
    """Re-read every .h5 and diff against the source. Returns list of problems."""
    import numpy as np

    problems = []

    # root + tf via _meta.h5
    with h5py.File(os.path.join(dst_dir, "_meta.h5"), "r") as f:
        for name in json.loads(f["root"].attrs["names"]):
            got = f["root"][_name_key(name)][()].tobytes()
            want = open(os.path.join(src_dir, name), "rb").read()
            if got != want:
                problems.append(f"root/{name}: {len(got)}B vs {len(want)}B mismatch")
        if "tf" in f:
            for rel in json.loads(f["tf"].attrs["relpaths"]):
                got = f["tf"][rel][()].tobytes()
                want = open(os.path.join(src_dir, "tf", rel), "rb").read()
                if got != want:
                    problems.append(f"tf/{rel}: mismatch")

    for ent in manifest:
        h5_path = os.path.join(dst_dir, ent["h5_relpath"])
        mod_dir = os.path.join(src_dir, ent["group"], ent["name"])
        with h5py.File(h5_path, "r") as f:
            for name in json.loads(f["files"].attrs["names"]):
                got = f["files"][_name_key(name)][()].tobytes()
                want = open(os.path.join(mod_dir, name), "rb").read()
                if got != want:
                    problems.append(f"{ent['group']}/{ent['name']}/{name}: mismatch")
            if "frames" in f:
                for key in f["frames"]:
                    g = f["frames"][key]
                    suffix = g.attrs["suffix"]
                    offs = g["offsets"][()]
                    pres = g["present"][()]
                    verb = g["verbatim"][()] if "verbatim" in g else None
                    blob = g["blob"]
                    for i in np.nonzero(pres)[0]:
                        got = blob[int(offs[i]):int(offs[i + 1])].tobytes()
                        srcf = os.path.join(mod_dir, f"{int(i):08d}{suffix}")
                        want = open(srcf, "rb").read()
                        if suffix == ".png" and verb is not None and not verb[i]:
                            a = cv2_imread_bytes(got)
                            b = cv2_imread_bytes(want)
                            if a is None or b is None or not np.array_equal(a, b):
                                problems.append(f"{ent['group']}/{ent['name']}/"
                                                f"{int(i):08d}.png: array mismatch")
                        elif got != want:
                            problems.append(f"{ent['group']}/{ent['name']}/"
                                            f"{int(i):08d}{suffix}: byte mismatch")
    return problems


def cv2_imread_bytes(b):
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", required=True, help="converted KITTI dataset tree")
    ap.add_argument("--dst_dir", required=True, help="output dir for the .h5 mirror")
    ap.add_argument("--workers", type=int, default=0,
                    help="modality pack workers (0 = cgroup CPU budget)")
    ap.add_argument("--verify", action="store_true",
                    help="re-read every .h5 against the source before success")
    ap.add_argument("--modalities", default="",
                    help="comma-separated group/name allowlist")
    args = ap.parse_args(argv)

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)
    if not os.path.isdir(src_dir):
        ap.error(f"--src_dir {src_dir} is not a directory")
    os.makedirs(dst_dir, exist_ok=True)

    allow = {m.strip() for m in args.modalities.split(",") if m.strip()}
    mods = [(g, n, d) for g, n, d in discover_modalities(src_dir)
            if not allow or f"{g}/{n}" in allow]
    if not mods:
        ap.error("no modality dirs discovered under --src_dir")

    cpus = args.workers or available_cpus()
    workers = max(1, min(cpus, len(mods)))
    encode_workers = max(2, cpus // 2)
    print(f"[pack] {src_dir} -> {dst_dir}")
    print(f"[pack] {len(mods)} modalities, {workers} workers "
          f"(x{encode_workers} encode threads), format v{FORMAT_VERSION}")

    tasks = [(g, n, d, os.path.join(dst_dir, g, f"{n}.h5"), encode_workers)
             for g, n, d in mods]

    manifest, failures = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_pack_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            g, n, out_h5, scan, dt, err = fut.result()
            if err:
                failures.append((f"{g}/{n}", err))
                print(f"[pack] FAIL {g}/{n}\n{err}", flush=True)
                continue
            rel = os.path.relpath(out_h5, dst_dir)
            ent = build_manifest_entry(g, n, scan, rel)
            manifest.append(ent)
            packed = os.path.getsize(out_h5)
            ratio = (packed / ent["src_bytes"]) if ent["src_bytes"] else 0
            print(f"[pack] {g}/{n}: {ent['src_file_count']} files "
                  f"{ent['src_bytes']/1e6:.1f}MB -> {packed/1e6:.1f}MB "
                  f"({ratio*100:.0f}%) {dt:.1f}s", flush=True)

    manifest.sort(key=lambda e: (e["group"], e["name"]))
    write_meta(src_dir, dst_dir, manifest)

    viz_src = os.path.join(src_dir, "viz")
    if os.path.isdir(viz_src):
        shutil.copytree(viz_src, os.path.join(dst_dir, "viz"), dirs_exist_ok=True)
        print(f"[pack] copied viz/ ({len(os.listdir(viz_src))} files)")

    src_files = sum(len(fs) for _, _, fs in os.walk(src_dir))
    dst_files = sum(len(fs) for _, _, fs in os.walk(dst_dir))
    src_bytes = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(src_dir) for f in fs)
    dst_bytes = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(dst_dir) for f in fs)
    print(f"[pack] done in {time.time()-t0:.0f}s: "
          f"{src_files} files / {src_bytes/1e9:.1f}GB -> "
          f"{dst_files} files / {dst_bytes/1e9:.1f}GB")

    if failures:
        print(f"[pack] {len(failures)} modality/-ies FAILED:")
        for m, _ in failures:
            print(f"    {m}")
        return 1

    if args.verify:
        print("[pack] verifying ...", flush=True)
        problems = verify(src_dir, dst_dir, manifest)
        if problems:
            print(f"[pack] VERIFY FAILED ({len(problems)} problems):")
            for p in problems[:50]:
                print(f"    {p}")
            return 2
        print("[pack] verify OK -- round-trip is lossless")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
