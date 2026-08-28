"""Shared spec for the packed-HDF5 mirror of a KITTI dataset tree.
"""
import json
import os
import re
import tempfile

import numpy as np

FORMAT_VERSION = "1.0"

# Files that sit directly in a modality dir under fixed names (not per-frame).
# Stored verbatim as gzipped bytes.
SIDECAR_NAMES = (
    "timestamps.txt",
    "interp_timestamps.txt",
    "errors.txt",
    "info.yaml",
    "data.txt",
    "interp_data.txt",
    "data.yaml",
)

# 8-digit frame index followed by a suffix. Group per-frame files by that suffix.
_PER_FRAME_RE = re.compile(r"^(\d{8})((?:\.[^.]+)|(?:_.+))$")

GZIP_LEVEL = 4
_INCOMPRESSIBLE_SUFFIXES = {".png", "_data.npz", "_data.hdf5"}


def _suffix_key(suffix):
    """'.png' -> 'png';  '_data.npz' -> 'data_npz';  '_metadata.yaml' -> 'metadata_yaml'."""
    return suffix.lstrip("._").replace(".", "_")


def _name_key(name):
    """'data.txt' -> 'data_txt' (a safe HDF5 dataset name)."""
    return name.replace(".", "_")


def parse_per_frame(filename):
    """Return ``(index:int, suffix:str)`` for a per-frame file, else ``None``."""
    m = _PER_FRAME_RE.match(filename)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def scan_modality(mod_dir):
    """Inventory one ``<group>/<name>/`` dir from what is on disk.

    Returns a dict:
      ``sidecars``   -> {filename: abspath}   (fixed-name files present)
      ``per_frame``  -> {suffix: {index: abspath}}
      ``n_frames``   -> int   (rows of timestamps.txt, else max frame index + 1)
      ``kind``       -> 'per_frame' | 'table' | 'raw' | 'empty'
      ``extra``      -> [filename, ...]   (anything unrecognised, packed verbatim)
    """
    sidecars, per_frame, extra = {}, {}, []
    for entry in sorted(os.listdir(mod_dir)):
        path = os.path.join(mod_dir, entry)
        if not os.path.isfile(path):
            continue
        pf = parse_per_frame(entry)
        if pf is not None:
            idx, suffix = pf
            per_frame.setdefault(suffix, {})[idx] = path
        elif entry in SIDECAR_NAMES:
            sidecars[entry] = path
        else:
            extra.append(entry)

    n_frames = 0
    ts = sidecars.get("timestamps.txt")
    if ts is not None:
        with open(ts, "r") as f:
            n_frames = sum(1 for line in f if line.strip())
    if n_frames == 0 and per_frame:
        n_frames = 1 + max(max(idxs) for idxs in per_frame.values())

    if per_frame:
        kind = "per_frame"
    elif "data.txt" in sidecars or "interp_data.txt" in sidecars:
        kind = "table"
    elif "data.yaml" in sidecars:
        kind = "raw"
    else:
        kind = "empty"

    return {"sidecars": sidecars, "per_frame": per_frame, "extra": extra,
            "n_frames": n_frames, "kind": kind}


# --------------------------------------------------------------------------- #
# low-level blob helpers
# --------------------------------------------------------------------------- #

def _gzip_kw(level):
    return {"compression": "gzip", "compression_opts": int(level)} if level else {}


def write_bytes_dataset(h5parent, name, data):
    """Store ``data`` (bytes) as a 1-D gzip-4 uint8 dataset."""
    arr = np.frombuffer(data, dtype="uint8")
    kw = {}
    if arr.size:
        kw = _gzip_kw(GZIP_LEVEL)
        kw["chunks"] = (min(arr.size, 8 << 20),)
    h5parent.create_dataset(name, data=arr, **kw)


def read_bytes_dataset(h5parent, name):
    return h5parent[name][()].tobytes()


def png_reencode(path):
    import cv2

    with open(path, "rb") as f:
        raw = f.read()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return raw, True
    try:
        ok, enc9 = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if ok and len(enc9) < len(raw):
            return enc9.tobytes(), False
    except Exception:
        pass
    return raw, True


def write_per_frame_group(h5parent, suffix, frame_paths, n_frames,
                          encode_workers=0):
    """Pack one suffix's per-frame files into ``h5parent/<suffix_key>``.
    """
    from concurrent.futures import ThreadPoolExecutor

    key = _suffix_key(suffix)
    is_png = suffix == ".png"
    incompressible = suffix in _INCOMPRESSIBLE_SUFFIXES

    present = np.zeros(n_frames, dtype=bool)
    verbatim = np.ones(n_frames, dtype=bool)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    g = h5parent.create_group(key)
    paths = [frame_paths.get(i) for i in range(n_frames)]

    spool = tempfile.TemporaryFile()
    try:
        if is_png:
            nthreads = encode_workers or available_cpus()
            pool = ThreadPoolExecutor(max_workers=max(1, nthreads))
            CHUNK = 512
            try:
                for base in range(0, n_frames, CHUNK):
                    hi = min(base + CHUNK, n_frames)
                    idxs = [i for i in range(base, hi) if paths[i] is not None]
                    encoded = dict(zip(
                        idxs, pool.map(png_reencode, [paths[i] for i in idxs])))
                    for i in range(base, hi):
                        if paths[i] is None:
                            offsets[i + 1] = offsets[i]
                            continue
                        out, verb = encoded[i]
                        verbatim[i] = verb
                        present[i] = True
                        spool.write(out)
                        offsets[i + 1] = offsets[i] + len(out)
            finally:
                pool.shutdown()
        else:
            for i in range(n_frames):
                if paths[i] is None:
                    offsets[i + 1] = offsets[i]
                    continue
                with open(paths[i], "rb") as f:
                    out = f.read()
                spool.write(out)
                present[i] = True
                offsets[i + 1] = offsets[i] + len(out)

        total = int(offsets[-1])
        kw = {}
        if total and not incompressible:
            kw = _gzip_kw(GZIP_LEVEL)
            kw["chunks"] = (min(total, 8 << 20),)
        blob = g.create_dataset("blob", shape=(total,), dtype="uint8", **kw)
        spool.seek(0)
        pos = 0
        step = 8 << 20
        while pos < total:
            chunk = spool.read(step)
            blob[pos:pos + len(chunk)] = np.frombuffer(chunk, dtype="uint8")
            pos += len(chunk)
    finally:
        spool.close()

    g.create_dataset("offsets", data=offsets, **_gzip_kw(GZIP_LEVEL))
    g.create_dataset("present", data=present, **_gzip_kw(GZIP_LEVEL))
    if is_png:
        g.create_dataset("verbatim", data=verbatim, **_gzip_kw(GZIP_LEVEL))
    g.attrs["suffix"] = suffix
    g.attrs["n_frames"] = int(n_frames)
    g.attrs["n_present"] = int(present.sum())


def restore_per_frame_group(h5group, out_dir, frame_slice=None):
    """Write ``{index:08d}<suffix>`` files from a packed per-frame group.

    ``frame_slice`` = ``(start, stop)`` restores only that half-open range.
    Returns the number of files written.
    """
    suffix = h5group.attrs["suffix"]
    n_frames = int(h5group.attrs["n_frames"])
    offsets = h5group["offsets"][()]
    present = h5group["present"][()]
    start, stop = (0, n_frames) if frame_slice is None else frame_slice
    start = max(0, start)
    stop = min(n_frames, stop)
    if stop <= start:
        return 0
    # one partial HDF5 read for the whole span
    span = h5group["blob"][int(offsets[start]):int(offsets[stop])]
    base = int(offsets[start])
    written = 0
    for i in range(start, stop):
        if not present[i]:
            continue
        lo = int(offsets[i]) - base
        hi = int(offsets[i + 1]) - base
        with open(os.path.join(out_dir, f"{i:08d}{suffix}"), "wb") as f:
            f.write(span[lo:hi].tobytes())
        written += 1
    return written


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def build_manifest_entry(group, name, scan, h5_relpath):
    src_bytes = 0
    src_files = 0
    for p in scan["sidecars"].values():
        src_bytes += os.path.getsize(p)
        src_files += 1
    for idxs in scan["per_frame"].values():
        for p in idxs.values():
            src_bytes += os.path.getsize(p)
            src_files += 1
    return {
        "group": group,
        "name": name,
        "kind": scan["kind"],
        "n_frames": scan["n_frames"],
        "h5_relpath": h5_relpath,
        "suffixes": sorted(_suffix_key(s) for s in scan["per_frame"]),
        "src_file_count": src_files,
        "src_bytes": src_bytes,
    }


def write_manifest(h5file, manifest):
    h5file.create_dataset("manifest", data=json.dumps(manifest, indent=1))
    h5file.attrs["format_version"] = FORMAT_VERSION


def read_manifest(h5file):
    raw = h5file["manifest"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def available_cpus():
    """cgroup-aware CPU count; falls back cleanly outside the ROS workspace."""
    try:
        from tartandriver_utils.os_utils import available_cpus as _ac
        return _ac()
    except Exception:
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            return os.cpu_count() or 1


__all__ = [
    "FORMAT_VERSION", "SIDECAR_NAMES",
    "parse_per_frame", "scan_modality",
    "write_bytes_dataset", "read_bytes_dataset",
    "write_per_frame_group", "restore_per_frame_group", "png_reencode",
    "build_manifest_entry", "write_manifest", "read_manifest",
    "available_cpus", "_suffix_key", "_name_key",
]
