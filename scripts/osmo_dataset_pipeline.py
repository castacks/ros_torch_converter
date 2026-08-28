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

from tartandriver_utils.os_utils import available_cpus, is_kitti_dir, load_yaml
from tartandriver_utils.rclone_stager import RcloneStager

from headless_bag_reannotation import (REANNOTATION_DEFAULTS, bag_has_topics,
                                       config_regenerated_topics, load_reannotation_settings,
                                       reannotate_bag, resolve_deploy_path)
from bag_message_report import build_report, write_report
from topic_sync_viz import render_sync_viz
from recover_truncated_mcap import recover_mcap


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERTER_SCRIPT = os.path.join(SCRIPT_DIR, "ros2bag_2_kitti_multiproc.py")
KITTI_2_HDF5_SCRIPT = os.path.join(SCRIPT_DIR, "kitti_2_hdf5.py")
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


STORAGE_ID_BY_EXT = {".mcap": "mcap", ".db3": "sqlite3"}


def repair_bag_storage_identifier(bag_dir, relpath=""):
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.exists(meta_path):
        return
    with open(meta_path, "r") as f:
        metadata = yaml.safe_load(f)
    info = metadata["rosbag2_bagfile_information"]
    if info.get("storage_identifier"):
        return
    exts = {os.path.splitext(p)[1] for p in info.get("relative_file_paths", [])}
    storage_ids = {STORAGE_ID_BY_EXT[e] for e in exts if e in STORAGE_ID_BY_EXT}
    if len(storage_ids) != 1:
        return  # ambiguous or unrecognized -- leave it for AnyReader to raise its own error
    info["storage_identifier"] = next(iter(storage_ids))
    print(f"[repair] {relpath}: metadata.yaml had empty storage_identifier -- "
          f"set to {info['storage_identifier']!r} from file extension", flush=True)
    with open(meta_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def bag_metadata_is_valid(bag_dir):
    """True iff bag_dir holds a metadata.yaml that parses into rosbag2 bag info."""
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0:
        return False
    try:
        with open(meta_path, "r") as f:
            metadata = yaml.safe_load(f)
    except yaml.YAMLError:
        return False
    return isinstance(metadata, dict) and "rosbag2_bagfile_information" in metadata


def reindex_bag(bag_dir, relpath="", recover_truncated=True):
    """Rebuild a bag's metadata.yaml from its storage files with `ros2 bag reindex`.

    Returns the list of (filename, action) tuples from any truncated-mcap
    recovery so the caller can record it; empty when nothing was recovered.
    """
    storage_files = sorted(f for f in os.listdir(bag_dir)
                           if os.path.splitext(f)[1] in STORAGE_ID_BY_EXT)
    # get rid of any 0-byte storage files, which ros2 bag reindex will choke on
    empty = [f for f in storage_files if os.path.getsize(os.path.join(bag_dir, f)) == 0]
    for name in empty:
        print(f"[reindex] {relpath}: dropping 0-byte {name}", flush=True)
        os.remove(os.path.join(bag_dir, name))
    storage_files = [f for f in storage_files if f not in empty]
    if not storage_files:
        raise RuntimeError("no non-empty storage files to reindex")

    recovery_actions = recover_mcap(bag_dir, relpath) if recover_truncated else []
    if recovery_actions:
        storage_files = sorted(f for f in os.listdir(bag_dir)
                               if os.path.splitext(f)[1] in STORAGE_ID_BY_EXT)
        if not storage_files:
            raise RuntimeError("no storage files left after truncated-mcap recovery")

    storage_ids = {STORAGE_ID_BY_EXT[os.path.splitext(f)[1]] for f in storage_files}
    if len(storage_ids) != 1:
        raise RuntimeError(f"mixed storage formats {sorted(storage_ids)}, "
                           "cannot pick a --storage-id for reindex")
    storage_id = next(iter(storage_ids))

    # A 0-byte metadata.yaml would be read as the bag's index; it holds nothing, so
    # remove it and let reindex write a fresh one in its place.
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    print(f"[reindex] {relpath}: rebuilding metadata.yaml from {len(storage_files)} "
          f"{storage_id} file(s) ...", flush=True)
    result = subprocess.run(["ros2", "bag", "reindex", bag_dir, "-s", storage_id],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"`ros2 bag reindex` exited with {result.returncode}:\n"
                           f"{result.stdout.strip()}")
    if not bag_metadata_is_valid(bag_dir):
        raise RuntimeError("`ros2 bag reindex` reported success but wrote no usable "
                           f"metadata.yaml:\n{result.stdout.strip()}")
    print(f"[reindex] {relpath}: metadata.yaml rebuilt", flush=True)
    return recovery_actions


def repair_bag_metadata(bag_dir, relpath="", recover_truncated=True):
    """Make a staged bag readable: rebuild metadata.yaml if it's missing or empty, then
    backfill an empty storage_identifier. Raises if the bag can't be repaired. Idempotent,
    so it's safe to call again on an already-repaired dir. Returns the truncated-mcap
    recovery actions (empty unless a reindex happened and recovered something)."""
    recovery_actions = []
    if not bag_metadata_is_valid(bag_dir):
        recovery_actions = reindex_bag(bag_dir, relpath, recover_truncated=recover_truncated)
    repair_bag_storage_identifier(bag_dir, relpath)
    return recovery_actions


def local_lsdir_recursive(root_dir):
    """Local-filesystem equivalent of RcloneStager.list(), for full-copy (data_dir) mode."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        relpath = os.path.relpath(dirpath, root_dir)
        for name in filenames:
            full = os.path.join(dirpath, name)
            entries.append({
                "IsDir": False,
                "Path": os.path.normpath(os.path.join(relpath, name)),
                "Size": os.path.getsize(full) if os.path.exists(full) else 0,
            })
    return entries


def split_exclusions(exclude_subdirs):
    """Split raw `exclude_subdirs` entries into (names, prefixes)."""
    names = {e for e in exclude_subdirs if "/" not in e}
    prefixes = {e.strip("/") for e in exclude_subdirs if "/" in e}
    return names, {p for p in prefixes if p}


def is_excluded(relpath, exclude_names, exclude_prefixes):
    """Matched prefix (or True for a name match) if `relpath` is excluded, else None."""
    if any(part in exclude_names for part in relpath.split("/")):
        return True
    return next((p for p in exclude_prefixes
                 if relpath == p or relpath.startswith(p + "/")), None)


def find_rosbag_run_dirs(root_dir, exclude_subdirs, limit_subfolder=None, stager=None, data_dir=None):
    """Find run-dirs under root_dir with >=1 non-empty .mcap, at any depth.

    Returns (run_dirs, needs_reindex): `needs_reindex` is the subset whose metadata.yaml
    is missing or 0-byte, which the caller must stage and `reindex_bag()` before use.
    This relaxes `is_rosbag_dir_filenames`, which requires metadata.yaml to be present --
    it can be rebuilt from the .mcap files, so its absence doesn't disqualify a run-dir.
    """
    entries = stager.list(root_dir) if stager else local_lsdir_recursive(os.path.join(data_dir, root_dir))

    sizes_by_dir = {}
    for entry in entries:
        if entry["IsDir"]:
            continue
        parent = os.path.dirname(entry["Path"])
        sizes_by_dir.setdefault(parent, {})[os.path.basename(entry["Path"])] = entry.get("Size", 0)

    exclude_names, exclude_prefixes = split_exclusions(exclude_subdirs)
    matched_prefixes = set()

    run_dirs, needs_reindex = [], set()
    for relpath, sizes in sizes_by_dir.items():
        if limit_subfolder is not None and relpath != limit_subfolder:
            continue
        excluded = is_excluded(relpath, exclude_names, exclude_prefixes)
        if excluded:
            if excluded is not True:
                matched_prefixes.add(excluded)
            continue
        if not any(name.endswith(".mcap") and size > 0 for name, size in sizes.items()):
            continue
        if not sizes.get("metadata.yaml", 0):
            needs_reindex.add(relpath)
        run_dirs.append(relpath)

    # A path exclusion is root-relative; one that matches nothing is usually a full
    # path pasted in by mistake, which would otherwise silently exclude nothing.
    unmatched = sorted(exclude_prefixes - matched_prefixes)
    if unmatched and limit_subfolder is None:
        print(f"[discovery] WARNING: path exclusion(s) matched no directory under "
              f"root_dir -- they must be relative to it: {unmatched}", flush=True)

    if needs_reindex:
        print(f"[discovery] {len(needs_reindex)} run-dir(s) have no usable metadata.yaml "
              f"(aborted recording) -- will reindex:", flush=True)
        for relpath in sorted(needs_reindex):
            print(f"    {relpath}", flush=True)

    return sorted(run_dirs), needs_reindex


def dataset_relpath(relpath):
    """Destination-side relpath for a run-dir: bags recorded as `<run>/rosbags/*.mcap`
    put their dataset at `<run>/`, not `<run>/rosbags/`"""
    parent, name = os.path.split(relpath.rstrip("/"))
    return parent if name == "rosbags" and parent else relpath


def filter_unconverted(run_dirs, dst_dir, stager=None, data_dir=None):
    """Drop run-dirs that already have a completed KITTI dataset at the destination."""
    if stager:
        return [relpath for relpath in run_dirs
                if not stager.exists(os.path.join(dst_dir, dataset_relpath(relpath)))]
    return [relpath for relpath in run_dirs
            if not is_kitti_dir(os.path.join(data_dir, dst_dir, dataset_relpath(relpath)))]


def classify_reannotation_needed(relpaths, root_dir, stager, data_dir, prestaged=None,
                                 annotation_topics=None, force=False):
    """Split `relpaths` into (needs_reannotation, unreadable): bags whose raw metadata.yaml
    is missing any of `annotation_topics` (the reannotate_config's `rosbag_record.topics`
    minus `reannotation.shared_topics`), and bags whose metadata.yaml couldn't be read at
    all. With `force` (reannotation.force) or no `annotation_topics` to check against 
    every readable bag needs reannotation regardless of what it already carries; the
    metadata is still read, so an unusable bag is caught here rather than mid-playback.
    Checked via metadata.yaml only (no full-bag download) so phase 1 only stages bags it
    will actually reannotate -- except for bags in `prestaged`, which the reindex phase
    already staged in full and whose rebuilt metadata is read straight off disk."""
    prestaged = prestaged or {}
    annotation_topics = list(annotation_topics or [])
    force = force or not annotation_topics
    needed, unreadable = [], []
    for relpath in relpaths:
        tmpdir = None
        try:
            if relpath in prestaged:
                bag_dir = prestaged[relpath]
            elif stager:
                tmpdir = tempfile.mkdtemp(prefix="osmo_pipeline_meta_")
                stager.copy_in(os.path.join(root_dir, relpath, "metadata.yaml"), tmpdir)
                bag_dir = tmpdir
            else:
                bag_dir = os.path.join(data_dir, root_dir, relpath)
            if force or not bag_has_topics(bag_dir, annotation_topics):
                needed.append(relpath)
        except Exception as e:
            # One unreadable metadata.yaml must not take down the whole batch.
            print(f"[reannotate] SKIP {relpath}: cannot read metadata.yaml: {e}", flush=True)
            unreadable.append((relpath, str(e)))
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
    return needed, unreadable


def reannotate_one(relpath, root_dir, kitti_out_root, orig_scratch_root, merged_scratch_root,
                   dst_dir, pipeline_config, reannotate_config, reannotate_settings,
                   domain_queue, stager, data_dir, dst_stager=None, prestaged_dir=None,
                   forced_topics=()):
    reannotated_dir = reannotate_settings["reannotated_dir"]
    domain_id = None
    try:
        if prestaged_dir:
            orig_dir = prestaged_dir
        elif stager:
            orig_dir = os.path.join(orig_scratch_root, relpath)
            os.makedirs(orig_dir, exist_ok=True)
            print(f"[reannotate] {relpath}: copying raw bag from {stager.remote} ...", flush=True)
            stager.copy_in(os.path.join(root_dir, relpath), orig_dir)
        else:
            orig_dir = os.path.join(data_dir, root_dir, relpath)
        repair_bag_metadata(orig_dir, relpath)

        if domain_queue is not None:
            domain_id = domain_queue.get()
        print(f"[reannotate] {relpath}: reannotating "
              f"(ROS_DOMAIN_ID={domain_id}) ...", flush=True)

        merged_root = os.path.join(merged_scratch_root, relpath)
        reann_kwargs = {"config_path": reannotate_config} if reannotate_config else {}
        merged_dir = reannotate_bag(orig_dir, merged_root, domain_id=domain_id,
                                    settings=reannotate_settings, **reann_kwargs)
        copy_bag_metadata(orig_dir, merged_dir)

        out_dir = os.path.join(kitti_out_root, dataset_relpath(relpath))
        os.makedirs(out_dir, exist_ok=True)
        print(f"[reannotate] {relpath}: writing message-count report + sync viz ...", flush=True)
        report_rows = build_report(orig_dir, merged_dir, forced_topics=forced_topics,
                                   remap_prefix=reannotate_settings["remap_prefix"])
        write_report(report_rows, os.path.join(out_dir, "reannotate_report"))
        try:
            render_sync_viz(merged_dir, resolve_config_path(pipeline_config),
                             os.path.join(out_dir, "topic_sync_viz.png"),
                             out_csv=os.path.join(out_dir, "topic_sync_viz.csv"))
        except Exception as e:
            print(f"[reannotate] {relpath}: topic_sync_viz failed (non-fatal): {e}", flush=True)

        if reannotated_dir:
            upload_stager = dst_stager or stager
            if upload_stager:
                bag_dst_relpath = os.path.join(dst_dir, reannotated_dir, relpath)
                print(f"[reannotate] {relpath}: copying reannotated bag to "
                      f"{upload_stager.remote}:{bag_dst_relpath} ...", flush=True)
                upload_stager.copy_out(merged_dir, bag_dst_relpath)
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
                local_out_dir, converter_workers, dst_stager=None, extra_dst_dir="",
                merged_bag_dir=None, orig_bag_dir=None, merged_scratch_dir=None,
                prestaged_dir=None, forced_topics=(), remap_prefix="",
                pack_hdf5=False, h5_out_root="", keep_kitti_local=True):
    scratch_dir = tempfile.mkdtemp(prefix="osmo_pipeline_raw_")
    out_dir = os.path.join(kitti_out_root, dataset_relpath(relpath))
    os.makedirs(out_dir, exist_ok=True)

    try:
        if merged_bag_dir:
            print(f"[convert] {relpath}: using bag reannotated in phase 1 ...", flush=True)
            src_dir = merged_bag_dir
        elif prestaged_dir:
            print(f"[convert] {relpath}: using bag staged + reindexed earlier in this batch ...", flush=True)
            src_dir = prestaged_dir
        elif stager:
            print(f"[convert] {relpath}: copying raw bag from {stager.remote} ...", flush=True)
            stager.copy_in(os.path.join(root_dir, relpath), scratch_dir)
            src_dir = scratch_dir
        else:
            src_dir = os.path.join(data_dir, root_dir, relpath)
        if not merged_bag_dir:  # merge_bags() output always has a valid storage_identifier
            repair_bag_metadata(src_dir, relpath)

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
        report_rows = build_report(report_original_dir, merged_bag_dir, kitti_dir=out_dir,
                                   forced_topics=forced_topics, remap_prefix=remap_prefix)
        write_report(report_rows, os.path.join(out_dir, "dataset_report"))

        primary_src = out_dir
        if pack_hdf5:
            primary_src = os.path.join(
                h5_out_root or (kitti_out_root.rstrip("/") + "_h5"),
                dataset_relpath(relpath))
            print(f"[convert] {relpath}: packing KITTI tree -> HDF5 mirror ...", flush=True)
            pack_rc = subprocess.run([
                sys.executable, KITTI_2_HDF5_SCRIPT,
                "--src_dir", out_dir, "--dst_dir", primary_src,
            ]).returncode
            if pack_rc != 0:
                return relpath, False, f"hdf5 pack exited with code {pack_rc}", None

        dst_relpath = os.path.join(dst_dir, dataset_relpath(relpath))
        upload_stager = dst_stager or stager
        copy_error = None
        if upload_stager:
            print(f"[convert] {relpath}: copying result to {upload_stager.remote}:{dst_relpath} ...", flush=True)
            try:
                upload_stager.copy_out(primary_src, dst_relpath)
            except subprocess.CalledProcessError as e:
                copy_error = f"primary copy-back to {upload_stager.remote}:{dst_relpath} failed: {e}"
                print(f"[convert] {relpath}: {copy_error}", flush=True)
        else:
            local_dst = os.path.join(data_dir, dst_relpath)
            print(f"[convert] {relpath}: copying result to {local_dst} ...", flush=True)
            shutil.copytree(primary_src, local_dst, dirs_exist_ok=True)

        if extra_dst_dir and stager:
            extra_relpath = os.path.join(extra_dst_dir, dataset_relpath(relpath))
            print(f"[convert] {relpath}: also copying result to {stager.remote}:{extra_relpath} ...", flush=True)
            try:
                stager.copy_out(out_dir, extra_relpath)
            except subprocess.CalledProcessError as e:
                extra_error = f"extra copy-back to {stager.remote}:{extra_relpath} failed: {e}"
                print(f"[convert] {relpath}: {extra_error}", flush=True)
                copy_error = f"{copy_error}; {extra_error}" if copy_error else extra_error

        if copy_error:
            return relpath, False, copy_error, None
        return relpath, True, None, kept_frames
    except subprocess.CalledProcessError as e:
        return relpath, False, str(e), None
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        # orig_bag_dir is only a scratch copy when a stager staged it (see
        # reannotate_one); in data_dir mode it's the caller's real source data.
        if orig_bag_dir and stager:
            shutil.rmtree(orig_bag_dir, ignore_errors=True)
        if prestaged_dir and stager and not merged_bag_dir:
            shutil.rmtree(prestaged_dir, ignore_errors=True)
        if merged_scratch_dir:
            shutil.rmtree(merged_scratch_dir, ignore_errors=True)
        if (not local_out_dir) or (pack_hdf5 and not keep_kitti_local):
            shutil.rmtree(out_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="root to scan for run-dirs (rclone-remote-relative, or data_dir-relative if --data_dir is set)")
    parser.add_argument("--dst_dir", required=True, help="destination for the final KITTI copy-back (rclone-remote-relative, or data_dir-relative if --data_dir is set)")
    parser.add_argument("--data_dir", default="", help="local mount root for full-copy mode (bulk_copy.sh); when set, reads/writes go straight to this mount instead of through an RcloneStager")
    parser.add_argument("--dst_remote", default="", help="rclone remote for the destination copy-back (dst_dir) and the already-converted check; defaults to the source remote (ignored in --data_dir mode)")
    parser.add_argument("--extra_dst_dir", default="", help="if set, also copy each converted KITTI dataset to this path on the *source* remote (stager) after the primary dst_dir copy-back -- e.g. mirror results back to the on-prem airlab_storage server in addition to --dst_remote/--dst_dir. Ignored in --data_dir mode.")
    parser.add_argument("--pipeline_config", required=True,
                        help="path to a ros_torch_converter config yaml")
    parser.add_argument("--limit_subfolder", default=None, help="restrict discovery to a single run-dir relpath")
    parser.add_argument("--local_out_dir", default="", help="if set, write each converted KITTI dataset to <local_out_dir>/<relpath> and keep it there (instead of a scratch tempdir deleted after copy-back), so the caller can read results locally. With pipeline.pack_hdf5 this is the HDF5-mirror root (and where _pipeline_summary.json lands); the KITTI tree goes under --kitti_local_dir.")
    parser.add_argument("--kitti_local_dir", default="", help="with pipeline.pack_hdf5: where the intermediate KITTI trees are written (default: a scratch tempdir removed at the end). Ignored unless pack_hdf5 is set.")
    parser.add_argument("--num_conversion_workers", type=int, default=None, help="bags to convert at once; overrides pipeline.num_conversion_workers")
    parser.add_argument("--num_reannotation_workers", type=int, default=None, help="bags to reannotate at once; overrides pipeline.num_reannotation_workers (default 1 -- one bag at a time so each live SLAM run gets the whole machine/GPU)")
    parser.add_argument("--phase_batch_size", type=int, default=None, help="bags per reannotate-then-convert batch; overrides pipeline.phase_batch_size (0/unset = one batch of everything pending)")
    parser.add_argument("--max_total_converter_workers", type=int, default=None, help="cap on converter pool workers across all bags; overrides pipeline.max_total_converter_workers (0 = use the cpu budget)")
    parser.add_argument("--exclude_subdirs", default=None, help="comma-separated exclusions for discovery: a bare name skips that dir at any depth, an entry with a '/' skips that root-relative path and everything under it; overrides pipeline.exclude_subdirs ('' / 'none' = exclude nothing)")
    parser.add_argument("--converter_extra_args", default=None, help="extra args passed through to the converter; overrides pipeline.converter_extra_args")
    parser.add_argument("--reannotate_config", default=None, help="playback config driving reannotation (repo-relative or absolute); overrides pipeline.reannotate_config")
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
    dst_stager = RcloneStager(remote=args.dst_remote) if (stager and args.dst_remote) else stager

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
    recover_truncated_mcap = bool(pcfg.get("recover_truncated_mcap", True))
    pack_hdf5 = bool(pcfg.get("pack_hdf5", False))
    keep_kitti_local = bool(pcfg.get("keep_kitti_local", True))
    reannotate_config_arg = pick(args.reannotate_config,
                                 pcfg.get("reannotate_config", "") or "",
                                 "reannotate_config") or ""

    exclude_subdirs_raw = str(pick(args.exclude_subdirs,
                                   pcfg.get("exclude_subdirs", "calibration"),
                                   "exclude_subdirs") or "")
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
    run_dirs, needs_reindex = find_rosbag_run_dirs(
        args.root_dir, exclude_subdirs, limit_subfolder=args.limit_subfolder,
        stager=stager, data_dir=args.data_dir)
    print(f"[discovery] found {len(run_dirs)} rosbag run-dir(s)")

    pending = filter_unconverted(run_dirs, args.dst_dir, stager=dst_stager, data_dir=args.data_dir)
    print(f"[discovery] {len(pending)} pending (not yet converted)")

    reannotate_config, reannotate_settings = "", dict(REANNOTATION_DEFAULTS)
    annotation_topics, forced_topics = [], []
    if reannotate:
        assert reannotate_config_arg, "pipeline.reannotate_config is required when pipeline.reannotate is true"
        reannotate_config = resolve_deploy_path(reannotate_config_arg)
        reannotate_settings = load_reannotation_settings(reannotate_config)
        reannotated_dir = reannotate_settings["reannotated_dir"]
        annotation_topics = config_regenerated_topics(reannotate_config, reannotate_settings)
        # tags the reports: which topics a bag that already had them got regenerated over
        forced_topics = annotation_topics if reannotate_settings["force"] else []
        if reannotate_settings["force"]:
            print(f"[reannotate] enabled (config={reannotate_config}), force=true -- every "
                  f"pending bag is reannotated, {num_reannotation_workers} at a time; bags "
                  f"that already have {annotation_topics} keep those copies under "
                  f"{reannotate_settings['remap_prefix']} and get fresh ones on the "
                  f"original names")
        elif annotation_topics:
            print(f"[reannotate] enabled (config={reannotate_config}) -- bags missing any of "
                  f"{annotation_topics} (rosbag_record.topics minus "
                  f"{reannotate_settings['shared_topics']}) will be reannotated first, "
                  f"{num_reannotation_workers} at a time")
        else:
            print(f"[reannotate] enabled (config={reannotate_config}) -- rosbag_record.topics "
                  f"is 'all', so there's nothing to test a bag against: every pending bag "
                  f"will be reannotated, {num_reannotation_workers} at a time")
        if reannotated_dir:
            print(f"[reannotate] merged reannotated bags will also be uploaded to "
                  f"{reannotated_dir}")

    orig_scratch_root = tempfile.mkdtemp(prefix="osmo_pipeline_orig_")
    merged_scratch_root = tempfile.mkdtemp(prefix="osmo_pipeline_merged_")
    if pack_hdf5:
        h5_out_root = args.local_out_dir or tempfile.mkdtemp(prefix="osmo_pipeline_h5scratch_")
        kitti_out_root = args.kitti_local_dir or tempfile.mkdtemp(prefix="osmo_pipeline_kittiscratch_")
    else:
        h5_out_root = ""
        kitti_out_root = args.local_out_dir or tempfile.mkdtemp(prefix="osmo_pipeline_kittiscratch_")
    print(f"[config] pack_hdf5={pack_hdf5}"
          + (f" (h5 -> {h5_out_root}, kitti -> {kitti_out_root}, "
             f"keep_kitti_local={keep_kitti_local})" if pack_hdf5 else ""))

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

        prestaged = {}
        batch_reindex = [relpath for relpath in batch if relpath in needs_reindex]
        for relpath in batch_reindex:
            reindex_start = time.time()
            if stager:
                bag_dir = os.path.join(orig_scratch_root, relpath)
                os.makedirs(bag_dir, exist_ok=True)
            else:
                bag_dir = os.path.join(args.data_dir, args.root_dir, relpath)
            try:
                if stager:
                    print(f"[reindex] {relpath}: copying raw bag from {stager.remote} ...", flush=True)
                    stager.copy_in(os.path.join(args.root_dir, relpath), bag_dir)
                recovery_actions = repair_bag_metadata(
                    bag_dir, relpath, recover_truncated=recover_truncated_mcap)
                if stager:
                    prestaged[relpath] = bag_dir
                if recovery_actions:
                    summary.append({
                        "relpath": relpath, "phase": "reindex",
                        "status": "recovered",
                        "reason": "; ".join(f"{n}: {w}" for n, w in recovery_actions),
                        "duration_s": round(time.time() - reindex_start, 1),
                    })
            except Exception as e:
                print(f"[reindex] FAIL {relpath}: {e}", flush=True)
                failed.append(relpath)
                summary.append({
                    "relpath": relpath, "phase": "reindex",
                    "status": "fail", "reason": str(e),
                    "duration_s": round(time.time() - reindex_start, 1),
                })
                if stager:
                    shutil.rmtree(bag_dir, ignore_errors=True)
        if batch_reindex:
            flush_summary()

        remaining = [relpath for relpath in batch if relpath not in failed]

        merged_by_relpath = {}
        if reannotate and remaining:
            needs_reannotation, unreadable = classify_reannotation_needed(
                remaining, args.root_dir, stager, args.data_dir, prestaged,
                annotation_topics=annotation_topics,
                force=reannotate_settings["force"])
            for relpath, err in unreadable:
                failed.append(relpath)
                summary.append({
                    "relpath": relpath, "phase": "reannotate",
                    "status": "fail", "reason": f"unreadable metadata.yaml: {err}",
                    "duration_s": 0.0,
                })
            if unreadable:
                flush_summary()
            print(f"[reannotate] {len(needs_reannotation)}/{len(remaining)} bag(s) in this "
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
                            args.pipeline_config, reannotate_config, reannotate_settings,
                            domain_queue, stager, args.data_dir, dst_stager,
                            prestaged.get(relpath), forced_topics,
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
                        dst_stager, args.extra_dst_dir, merged_dir, orig_dir,
                        merged_scratch_dir, prestaged.get(relpath),
                        forced_topics, reannotate_settings["remap_prefix"],
                        pack_hdf5, h5_out_root, keep_kitti_local,
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
    if kitti_out_root not in (args.local_out_dir, args.kitti_local_dir):
        shutil.rmtree(kitti_out_root, ignore_errors=True)
    if pack_hdf5 and h5_out_root and h5_out_root != args.local_out_dir:
        shutil.rmtree(h5_out_root, ignore_errors=True)

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
