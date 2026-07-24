import os
import yaml
import argparse
import copy
import json
import time
from datetime import timedelta
import multiprocessing
from multiprocessing import Pool, Manager, set_start_method
from threading import Thread
import sys

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tabulate import tabulate

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.os_utils import available_cpus, load_yaml

from ros_torch_converter.converter import str_to_cvt_class
from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.datatypes.base import TimeSpec
from ros_torch_converter.datatypes.intrinsics import CameraInfoTorch

"""
Script to create kitti-formatted datasets from ros2 bags
General algo:
    1. Identify missing topics
    2. Do a pass through to figure out all the target times
    3. Set up TFs
    4. Convert messages to KITTI format & interpolate interpolable types
    5. Verify
"""

RESET = '\033[0m'
GREEN = '\033[92m'
GRAY = '\033[90m'
RED = '\033[91m'
YELLOW = '\033[33m'

GROUP_COLORS = {
    'autonomy':       '\033[93m',  # Bright yellow
    'controls':       '\033[96m',  # Bright cyan
    'sensors':        '\033[94m',  # Bright blue
    'super_odometry': '\033[95m',  # Bright magenta
}

def group_color(group):
    return GROUP_COLORS.get(group, GRAY)

def apply_color(color, text):
    return f"{color}{text}{RESET}" if color else text

def setup_queue(reader, topics, dt):
    """
    Initialize message queues based on config
    """
    start_time = reader.start_time * 1e-9
    end_time = reader.end_time * 1e-9

    target_times = np.arange(start_time, end_time, dt)

    queue = {
        'target_times': target_times,
        'topic_times': {},
        'topic_error': {},
    }

    for topic in topics:
        queue['topic_times'][topic] = -np.ones(len(target_times))
        queue['topic_error'][topic] = float('inf') * np.ones(len(target_times))

    return queue

def get_filtered_config(connections, cvt_info):
    """
    Check all required topics in the config are present in connections.
    
    Returns:
        filtered_config: in format {'topic': {kwargs}, ...}.
    """
    connection_topics = [x.topic for x in connections]
    missing_topics = []
    missing_optional_topics = []
    filt_cvt_info = {} # cvt_info with only topics present in bag
    for k, ci in cvt_info.items():
        topic = ci['topic']
        if topic not in connection_topics:
            if not ci['optional']:
                missing_topics.append(topic)
            else:
                missing_optional_topics.append(topic)
        else:
            filt_cvt_info[k] = copy.deepcopy(ci)

    assert not missing_topics, "Bag missing config required topics:{}".format("\n\t".join(missing_topics))

    missing_optional = '\n'.join(missing_optional_topics)
    print(f"\nMissing optional topics:\n{missing_optional}")

    return filt_cvt_info

def check_missing_types(config):
    """
    Check all types listed in the config exist
    """
    missing_types = []
    for topic_data in config["topics"]:
        msg_type = topic_data["type"]
        if msg_type not in str_to_cvt_class:
            missing_types.append((topic_data["topic"], msg_type))

    if missing_types:
        print("\nERROR: Missing converters for the following message types:")
        for topic, msg_type in missing_types:
            print(f"  Topic: {topic}")
            print(f"  Type: {msg_type}")
        print("\nAvailable converters:")
        for msg_type in sorted(str_to_cvt_class.keys()):
            print(f"  - {msg_type}")
        print("\nPlease add the missing types to str_to_cvt_class in converter.py")
        exit(1)

    print("All message types have converters available ✓")


def display_progress_monitor(progress_queue, topic_list, shard_counts, bag, use_color=False):
    """
    Monitor and display progress in a live-updating dashboard style.
    Supports sharded topics: multiple workers reporting to the same ckey are aggregated.
    """
    # shards[ckey][shard_id] = {status, progress, total, fps}
    shards = {
        ckey: {i: {'status': 'waiting', 'progress': 0, 'total': 0, 'fps': 0.0}
               for i in range(shard_counts[ckey])}
        for ckey in topic_list
    }

    def agg(ckey):
        """Aggregate shard statuses into a single display status for a ckey."""
        shard_list = list(shards[ckey].values())
        statuses = [s['status'] for s in shard_list]
        progress = sum(s['progress'] for s in shard_list)
        total    = sum(s['total']    for s in shard_list)
        fps      = sum(s['fps']      for s in shard_list)
        if all(s == 'completed' for s in statuses):
            status = 'completed'
        elif any(s == 'error' for s in statuses):
            status = 'error'
        elif any(s in ('processing', 'completed') for s in statuses):
            status = 'processing'
        else:
            status = 'waiting'
        return status, progress, total, fps

    def print_dashboard():
        sys.stdout.write('\033[2J\033[H')

        print('4. Converting to KITTI')
        print(f"bag: {bag}")
        print(f"{'Topic':<50} {'Status':<15} {'Progress':<25} {'Speed':>10}")
        print("-" * 100)

        for ckey in topic_list:
            agg_status, progress, total, fps = agg(ckey)

            if agg_status == 'completed':
                status_str = f"{GREEN}✓ COMPLETE     {RESET}" if use_color else f"{'✓ COMPLETE':<15}"
                bar = "█" * 20
            elif agg_status == 'error':
                status_str = f"{'✗ ERROR':<15}"
                bar = " " * 20
            elif agg_status == 'processing':
                status_str = f"{'Processing':<15}"
                pct = min(progress / total, 1.0) if total > 0 else 0
                filled = int(20 * pct)
                bar = "█" * filled + "░" * (20 - filled)
            else:
                status_str = f"{GRAY}Waiting        {RESET}" if use_color else f"{'Waiting':<15}"
                bar = "░" * 20

            n = shard_counts[ckey]
            shard_tag = f" [{n}x]" if n > 1 else ""
            progress_str = f"{progress}/{total}"
            fps_str = f"{fps:>6.1f} fps"

            if use_color:
                color = group_color(ckey.split('/')[0])
                topic_display = f"{color}{ckey + shard_tag:<50}{RESET}"
            else:
                topic_display = f"{ckey + shard_tag:<50}"
            print(f"{topic_display} {status_str} [{bar}] {progress_str:>10} {fps_str:>10}")

        sys.stdout.flush()

    print_dashboard()

    while True:
        try:
            try:
                ckey, shard_id, status, progress, total, fps = progress_queue.get(timeout=0.5)
                shards[ckey][shard_id] = {'status': status, 'progress': progress, 'total': total, 'fps': fps}
                print_dashboard()
            except:
                all_done = all(agg(ckey)[0] in ('completed', 'error') for ckey in topic_list)
                if all_done:
                    break
        except KeyboardInterrupt:
            break

    print_dashboard()

def process_cvt_entry_wrapper(args_tuple):
    """
    Wrapper for multiprocessing. Processes a single cvt_info entry.
    For INTERP topics, collects all messages and calls to_interp at the end.
    shard_id/frame_offset support splitting a topic across multiple workers by frame range:
      topic_times is the shard's slice; frame_offset shifts local indices to global ones.
    """
    bagpath, ckey, cinfo, parsed_args, topic_times, camera_info_torch, n_frames, is_interp, shard_id, frame_offset, progress_queue = args_tuple

    topic = cinfo['topic']
    base_dir = cinfo['dir']
    msgtype = cinfo['msgtype']

    progress_queue.put((ckey, shard_id, 'starting', 0, 0, 0.0))

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    checks = []

    # note that behavior is non-deterministic if a topic has multiple msgs with the same timestamp
    try:
        with AnyReader([bagpath], default_typestore=typestore) as reader:
            matching_connections = [x for x in reader.connections if x.topic == topic]

            if not matching_connections:
                progress_queue.put((ckey, shard_id, 'error', 0, 0, 0.0))
                return ckey, shard_id, checks, 0

            torch_dtype = str_to_cvt_class[msgtype]
            start = time.time()
            last_idx = -1
            processed_count = 0
            interp_buf = []

            for conn, timestamp, rawdata in reader.messages(connections=matching_connections):
                try:
                    msg = reader.deserialize(rawdata, conn.msgtype)
                except Exception:
                    continue

                msg_time = timestamp * 1e-9
                if hasattr(msg, "header") and not parsed_args.use_bag_time:
                    stamp_time = stamp_to_time(msg.header.stamp)
                    if stamp_time != 0:
                        msg_time = stamp_time

                # Collect all messages for INTERP topics
                if is_interp:
                    torch_data = torch_dtype.from_rosmsg(msg)
                    torch_data.stamp = msg_time
                    if (len(interp_buf) == 0) or (msg_time - interp_buf[-1].stamp > 1e-16):
                        interp_buf.append(torch_data)

                # Match against this shard's slice of target times
                target_diffs = np.abs(topic_times - msg_time)
                local_idxs = np.argwhere(target_diffs < 1e-16).flatten()

                if len(local_idxs) > 0:
                    last_idx = max(last_idx, local_idxs[0].item())
                    processed_count += 1
                    dur = time.time() - start
                    rate = last_idx / dur if dur > 0 else 0

                    if processed_count % 50 == 0 or processed_count == n_frames:
                        progress_queue.put((ckey, shard_id, 'processing', processed_count, n_frames, rate))

                    # Track global indices for verification
                    global_idxs = local_idxs + frame_offset
                    checks.append(global_idxs)

                    try:
                        if camera_info_torch is not None:
                            torch_data = torch_dtype.from_rosmsg(msg, camera_info_torch=camera_info_torch, rectify=True)
                        else:
                            torch_data = torch_dtype.from_rosmsg(msg)

                        for local_idx, global_idx in zip(local_idxs, global_idxs):
                            if parsed_args.fill_missing_stamps and torch_data.stamp == -1:
                                torch_data.stamp = topic_times[local_idx]
                            torch_data.to_kitti(base_dir, global_idx)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue

        # Call to_interp after full message pass for INTERP topics
        if is_interp:
            torch_dtype.to_interp(base_dir, interp_buf)

        dur = time.time() - start
        rate = n_frames / dur if dur > 0 else 0
        progress_queue.put((ckey, shard_id, 'completed', n_frames, n_frames, rate))

        return ckey, shard_id, checks, len(interp_buf)

    except Exception:
        import traceback
        traceback.print_exc()
        progress_queue.put((ckey, shard_id, 'error', 0, n_frames, 0.0))
        return ckey, shard_id, [], 0

if __name__ == '__main__':
    # Use 'spawn' instead of 'fork' to avoid issues with PyTorch/OpenCV in forked processes
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # already set

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config')
    parser.add_argument('--calib_file', type=str, required=False, help='overwrite some tfs with calibs from this file')
    parser.add_argument('--src_dir', type=str, required=True, help='path to input dir')
    parser.add_argument('--dst_dir', type=str, required=True, help='path to output dir')
    parser.add_argument('--dryrun', action='store_true', help='set this flag to check data w/o parsing it')
    parser.add_argument('--no_plot', action='store_true', help='set this flag to not display the plot')
    parser.add_argument('--force', action='store_true', help='dont ask to overwrite')
    parser.add_argument('--use_bag_time', action='store_true', help='set this flag to use bag time for all stamps (not recommended)')
    parser.add_argument('--fill_missing_stamps', action='store_true', help='set this flag to use bag time for any data which does not have stamps')
    parser.add_argument('--skip_tf', action='store_true', help='set this flag to skip TF processing (useful if TF tree is broken)')
    parser.add_argument('--rectify', action='store_true', help='set this flag to rectify compressed images using camera_info (requires camera_info topics in bag)')
    parser.add_argument('--num_workers', type=str, default=None, help='number of parallel workers (default: min of entries and CPU cores, or "max" to use all CPU cores)')
    parser.add_argument('--color', action='store_true', help='use colored output for different topics')
    parser.add_argument('--no_render_video', action='store_true', help='skip video rendering after conversion')
    parser.add_argument('--video_config', type=str, default=None, help='path to a video render config yaml (HUD + picture-in-picture insets); see config/video/*.yaml')
    args = parser.parse_args()

    if os.path.exists(args.dst_dir) and not args.force:
        x = input('\n{} exists. Overwrite? [Y/n] '.format(args.dst_dir))
        if x == 'n':
            exit(0)

    config = load_yaml(args.config)

    cvt_info = {}
    for tconf in config['topics']:
        name = f"{tconf['group']}/{tconf['name']}"
        assert name not in cvt_info.keys()

        is_interp = str_to_cvt_class[tconf['type']].time_spec == TimeSpec.INTERP
        n_parallel = tconf.get('parallel_workers', 1)
        assert not (is_interp and n_parallel > 1), \
            f"Topic {tconf['topic']} is INTERP and cannot use parallel_workers > 1 (For now)" # TODO (low priority) move interp to separate job after main to_kitti is done
        cvt_info[name] = {
            'name': tconf['name'],
            'group': tconf['group'],
            'topic': tconf['topic'],
            'msgtype': tconf['type'],
            'interp': is_interp,
            'optional': tconf.get('optional', False),
            'parallel_workers': n_parallel,
            'dir': os.path.join(args.dst_dir, tconf['group'], tconf['name'])
        }

    all_topics = set([x['topic'] for x in cvt_info.values()])

    # Check for missing message type converters upfront
    print('\n1. Check Topics in Bag:')
    check_missing_types(config)

    bag_fps = sorted([x for x in os.listdir(args.src_dir) if '.mcap' in x])

    print('processing these bags:')
    for bfp in bag_fps:
        print('\t' + bfp)

    bagpath = Path(args.src_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    ##print proc check table
    tabdata = [['Name', 'Group', 'Msg Type', 'Interp', 'Optional', 'Topic', 'Save Path']]
    for _, _ci in sorted(cvt_info.items()):
        row = [_ci[k] for k in ['name', 'group', 'msgtype', 'interp', 'optional', 'topic']]
        row.append("{dst_dir}"+_ci['dir'].split(args.dst_dir)[1])
        if args.color:
            color = group_color(_ci['group'])
            row = [f"{color}{cell}{RESET}" for cell in row]
        tabdata.append(row)

    print(f"\ndst_dir: {args.dst_dir}")
    print(tabulate(tabdata, headers='firstrow', tablefmt='github'))

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        all_connections = [x for x in reader.connections if x.msgcount > 0 and x.topic in all_topics]
        filt_cvt_info = get_filtered_config(all_connections, cvt_info)
        target_topics = set([x['topic'] for x in filt_cvt_info.values()])

    if not args.force:
        x = input('\nDoes this look correct? [Y/n] ')
        if x == 'n':
            exit(0)
        print()

    full_start = time.time()
    print('\n2. Check timestamps')
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        # Do not add topics with 0 count to the queue, else sync issues
        connections = [x for x in reader.connections if x.msgcount > 0 and x.topic in target_topics]
        topic_msgcount = {x.topic: x.msgcount for x in connections}

        queue = setup_queue(reader, list(target_topics), config['dt'])
        topic_frame = {}

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if hasattr(msg, "header") and not args.use_bag_time:
                stamp_time = stamp_to_time(msg.header.stamp)
                if stamp_time != 0:
                    msg_time = stamp_time
            else:
                msg_time = timestamp * 1e-9

            # first-seen frame wins per topic
            if topic not in topic_frame:
                if hasattr(msg, "child_frame_id"):
                    topic_frame[topic] = msg.child_frame_id
                elif hasattr(msg, "header"):
                    topic_frame[topic] = msg.header.frame_id

            tdiffs = queue['target_times'] - msg_time

            better_mask = np.abs(tdiffs) < queue['topic_error'][topic]

            if not config['backward_interpolation']:
                better_mask = better_mask & (tdiffs >= 0.)

            queue['topic_error'][topic][better_mask] = np.abs(tdiffs)[better_mask]
            queue['topic_times'][topic][better_mask] = msg_time

    has_calib_file = False
    if 'calibration' in config.keys():
        print('applying calib file from config...')
        calib_config = config['calibration']
        has_calib_file = True

    elif args.calib_file is not None:
        print('applying calib file from cli...')
        calib_config = load_yaml(args.calib_file)
        has_calib_file = True

    if not has_calib_file:
        print('no calib file provided. Note that for Yamaha data this is probably wrong!')

    if args.skip_tf:
        print('Skipping TF processing as requested...')
        tf_manager = None
    else:
        print('\n3. Handle TF')
        tf_manager = TfManager.from_rosbag(bagpath, device='cuda')

        if has_calib_file:
            tf_manager.update_from_calib_config(calib_config)

    ##  do some proc to get consecutive segments
    all_valid_mask = np.ones(len(queue['target_times']), dtype=bool)

    optional_topics = {ci['topic'] for ci in filt_cvt_info.values() if ci['optional']}

    topic_valid_masks = {}
    topic_coverage = {}
    for topic, err in queue['topic_error'].items():
        topic_mask = err < config['interp_tol']
        topic_valid_masks[topic] = topic_mask
        topic_coverage[topic] = float(topic_mask.mean())
        if topic not in optional_topics:
            all_valid_mask = all_valid_mask & topic_mask

    # Each topic's tf validity is gated against its own frame's ancestor chain,
    topic_tf_ranges = {}
    tf_valid_mask = np.ones(len(queue['target_times']), dtype=bool)
    for topic, times in queue['topic_times'].items():
        if tf_manager is None:
            tmin, tmax = -np.inf, np.inf
        else:
            tmin, tmax = tf_manager.tf_tree.get_valid_time_range(topic_frame.get(topic))
        topic_tf_ranges[topic] = (tmin, tmax)
        tf_valid_mask = tf_valid_mask & (times > tmin) & (times < tmax)
    all_valid_mask = all_valid_mask & tf_valid_mask

    tf_tmin = min((r[0] for r in topic_tf_ranges.values()), default=-np.inf)
    tf_tmax = max((r[1] for r in topic_tf_ranges.values()), default=np.inf)

    if not all_valid_mask.any():
        print(f"{RED}topics not sync'ed! cause:{RESET}")
        for topic, mask in topic_valid_masks.items():
            if topic in optional_topics:
                continue  # optional topics never gate all_valid_mask
            if not mask.any():
                min_err = np.min(queue['topic_error'][topic])
                print(f"  {topic}: never within interp_tol={config['interp_tol']} (min error={min_err:.4f}s)")
        for topic, (tmin, tmax) in topic_tf_ranges.items():
            times = queue['topic_times'][topic]
            if not ((times > tmin) & (times < tmax)).any():
                print(f"  {topic}: frame '{topic_frame.get(topic)}' tf valid range [{tmin:.3f}, {tmax:.3f}] never overlaps its matched message times")

        assert False, "topics not sync'ed! see cause(s) above"

    total_slots = all_valid_mask.shape[0]
    print("\nPer-topic frame coverage (optional topics are best-effort synced)")
    for topic, cov in sorted(topic_coverage.items(), key=lambda kv: kv[1]):
        kind = "optional" if topic in optional_topics else "required"
        kept = int(topic_valid_masks[topic].sum())
        print(f"  {topic}: {kept}/{total_slots} ({cov * 100:.1f}%, {kind}, "
              f"{topic_msgcount.get(topic, 0)} msgs)")

    queue['target_times'] = queue['target_times'][all_valid_mask]

    for topic in queue['topic_times'].keys():
        times = queue['topic_times'][topic][all_valid_mask]
        valid = topic_valid_masks[topic][all_valid_mask]
        queue['topic_times'][topic] = np.where(valid, times, np.nan)
        queue['topic_error'][topic] = queue['topic_error'][topic][all_valid_mask]

    n_frames = int(all_valid_mask.sum())
    print('keeping {}/{} potential frames.'.format(n_frames, all_valid_mask.shape[0]))

    os.makedirs(args.dst_dir, exist_ok=True)
    np.savetxt(os.path.join(args.dst_dir, 'target_timestamps.txt'), queue['target_times'])

    ## setup folder structure/populate timestamps
    for cvt_config in filt_cvt_info.values():
        topic = cvt_config['topic']
        topic_dir = cvt_config['dir']
        os.makedirs(topic_dir, exist_ok=True)

        np.savetxt(os.path.join(topic_dir, 'timestamps.txt'), queue['topic_times'][topic])
        np.savetxt(os.path.join(topic_dir, 'errors.txt'), queue['topic_error'][topic])

    if tf_manager is not None:
        tf_manager.to_kitti(args.dst_dir)

        print('TF TREE:\n')
        print(tf_manager.tf_tree)

    plt.plot(queue['target_times'], marker='.', label='target_times')

    x = np.arange(len(queue['target_times']))

    if tf_manager is not None:
        if tf_tmin > 0.:
            idx = queue['target_times'][queue['target_times'] > tf_tmin].argmin()
            plt.axvline(idx, color='r', label='Tf tmin (idx {})'.format(idx))

        if tf_tmax < 1e16:
            idx = queue['target_times'][queue['target_times'] < tf_tmax].argmax()
            plt.axvline(idx, color='r', label='Tf tmax (idx {})'.format(idx))

    for topic in target_topics:
        times = queue['topic_times'][topic]
        error = queue['topic_error'][topic]
        mask = error < config['interp_tol']
        plt.plot(x[mask], queue['topic_times'][topic][mask], marker='.', label="{} ({} bad)".format(topic, len(mask) - mask.sum()))

    plt.title('time sync graph')
    plt.legend()

    plt.savefig(os.path.join(args.dst_dir, 'sync_plot.png'), dpi=300)

    #save tf
    if tf_manager is not None:
        tf_manager.to_kitti(args.dst_dir)

    if args.dryrun:
        if not args.no_plot:
            plt.show()
        exit(0)

    # Cache camera_info messages if rectification is requested
    camera_info_cache = {}
    if args.rectify:
        with AnyReader([bagpath], default_typestore=typestore) as reader:
            camera_info_topics = [topic for topic in target_topics if 'camera_info' in topic]
            if camera_info_topics:
                print(f"Caching camera_info from {len(camera_info_topics)} topics for rectification...")
                camera_info_connections = [x for x in reader.connections if x.topic in camera_info_topics]
                for connection, timestamp, rawdata in reader.messages(connections=camera_info_connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    camera_info_torch = CameraInfoTorch.from_rosmsg(msg, device='cpu')
                    camera_info_cache[connection.topic] = camera_info_torch
                print(f"Cached {len(camera_info_cache)} camera_info messages")
            else:
                print("WARNING: --rectify flag set but no camera_info topics found in bag!")

    # Process cvt_info entries in parallel using multiprocessing
    progress_queue = Manager().Queue()

    process_args = []
    shard_counts = {}
    for ckey, cinfo in sorted(filt_cvt_info.items()):
        topic = cinfo['topic']
        topic_times_full = queue['topic_times'][topic]
        is_interp = cinfo['interp']
        n_parallel = cinfo['parallel_workers']

        camera_info_torch = None
        if args.rectify and cinfo['msgtype'] == 'CompressedImage':
            base_topic = topic.replace('/image_raw/compressed', '').replace('/image/compressed', '').replace('/compressed', '')
            camera_info_topic = base_topic + '/camera_info'
            camera_info_torch = camera_info_cache.get(camera_info_topic, None)
            if camera_info_torch is None:
                print(f"\nWARNING: No camera_info found for {topic}, skipping rectification")

        shard_counts[ckey] = n_parallel
        if n_parallel > 1:
            chunks = np.array_split(topic_times_full, n_parallel)
            frame_offset = 0
            for shard_id, chunk in enumerate(chunks):
                process_args.append((bagpath, ckey, cinfo, args, chunk, camera_info_torch, len(chunk), is_interp, shard_id, frame_offset, progress_queue))
                frame_offset += len(chunk)
        else:
            process_args.append((bagpath, ckey, cinfo, args, topic_times_full, camera_info_torch, n_frames, is_interp, 0, 0, progress_queue))

    total_workers_needed = sum(shard_counts.values())
    cpu_budget = available_cpus()
    if args.num_workers is None:
        num_workers = min(total_workers_needed, cpu_budget)
    elif args.num_workers == "max":
        num_workers = cpu_budget
    else:
        num_workers = min(cpu_budget, int(args.num_workers))

    tokitti_start = time.time()
    print(f'\n[stage] +{timedelta(seconds=int(time.time() - full_start))} 4. Converting to KITTI')
    print(f"\nProcessing {total_workers_needed} workers ({len(filt_cvt_info)} topics) in parallel using {num_workers} pool workers (cpu budget {cpu_budget})...\n")

    monitor_thread = Thread(target=display_progress_monitor, args=(progress_queue, sorted(filt_cvt_info.keys()), shard_counts, args.src_dir, args.color))
    monitor_thread.daemon = True
    monitor_thread.start()

    with Pool(processes=num_workers) as pool:
        try:
            async_result = pool.map_async(process_cvt_entry_wrapper, process_args)
            results = async_result.get(timeout=7200)  # 2 hour timeout
            # multiprocessing raises multiprocessing.TimeoutError here, which is NOT a
        except multiprocessing.TimeoutError:
            print("\nERROR: Processing timed out after 2 hours")
            pool.terminate()
            pool.join()
            raise

    monitor_thread.join(timeout=2)

    # Collect results, merging checks by topic across shards
    checks = {}
    interp_counts = {}
    for ckey, shard_id, topic_checks, interp_count in results:
        topic = filt_cvt_info[ckey]['topic']
        if topic not in checks:
            checks[topic] = []
        checks[topic].extend(topic_checks)
        if interp_count > 0:
            interp_counts[topic] = interp_count

    full_dur = time.time() - full_start
    tokitti_dur = time.time() - tokitti_start
    rate = n_frames / tokitti_dur if tokitti_dur > 0 else 0
    print(f'\n[stage] +{timedelta(seconds=int(full_dur))} 5. Verification:')
    print(f"{len(filt_cvt_info)}/{len(cvt_info)} present entries")
    print(f"All present entries completed successfully")
    print(f'Total processing time: {timedelta(seconds=int(full_dur))}')
    print(f'to_kitti time: {timedelta(seconds=int(tokitti_dur))}')
    print(f'to_kitti rate: {rate:.1f} frames/sec')

    ## check that all idxs got filled
    checks = {k: np.sort(np.concatenate(v)) if len(v) > 0 else np.array([]) for k, v in checks.items()}

    print("{}/{} valid frames for dataset".format(all_valid_mask.sum(), all_valid_mask.shape[0]))

    topic_summary = {}
    for ckey, cinfo in sorted(cvt_info.items()):
        topic = cinfo['topic']
        if ckey not in filt_cvt_info:
            topic_summary[topic] = {'frames_written': 0, 'status': 'SKIPPED'}
            continue
        idxs = checks.get(topic, np.array([]))
        frames_written = len(np.unique(idxs)) if len(idxs) > 0 else 0
        complete = frames_written > 0 and all(np.unique(idxs) == np.arange(all_valid_mask.sum()))
        if complete:
            status = 'SUCCESS'
        elif frames_written > 0 and topic in optional_topics:
            status = 'PARTIAL'
        else:
            status = 'FAIL'
        topic_summary[topic] = {'frames_written': frames_written, 'status': status}

    rows = []
    status_str = {
        'SUCCESS': "SUCCESS ✓",
        'PARTIAL': "PARTIAL ~",
        'FAIL': "FAIL ✗",
        'SKIPPED': "SKIPPED -",
    }
    status_clr = {
        'SUCCESS': GREEN if args.color else '',
        'PARTIAL': YELLOW if args.color else '',
        'FAIL': RED if args.color else '',
        'SKIPPED': GRAY if args.color else '',
    }

    for ckey in sorted(cvt_info.keys()):
        cinfo = cvt_info[ckey]
        topic = cinfo['topic']
        grp_clr = group_color(cinfo['group']) if args.color else ''

        if ckey not in filt_cvt_info:
            rows.append([
                apply_color(status_clr['SKIPPED'], status_str['SKIPPED']),
                apply_color(grp_clr, topic),
                apply_color(grp_clr, '- (optional)'),
                apply_color(grp_clr, '-'),
            ])
            continue

        info = topic_summary[topic]
        status = apply_color(status_clr[info['status']], status_str[info['status']])
        if info['frames_written'] > 0:
            frames_str = f"{info['frames_written']}/{all_valid_mask.sum()}"
        else:
            frames_str = f"0/{all_valid_mask.sum()} (NO DATA)"

        interp_str = str(interp_counts[topic]) if topic in interp_counts else '-'
        rows.append([
            status,
            apply_color(grp_clr, topic),
            apply_color(grp_clr, frames_str),
            apply_color(grp_clr, interp_str)
        ])
    print(tabulate(rows, headers=['Status', 'Topic', 'Frames', 'Interp Samples'], tablefmt='github'))

    # Persist the numbers above so they survive OSMO log truncation (see
    # bag_message_report.py, which reads this back in to build the combined
    # bag-vs-KITTI report).
    conversion_stats = {
        'dt': config['dt'],
        'interp_tol': config['interp_tol'],
        'potential_frames': int(all_valid_mask.shape[0]),
        'kept_frames': int(all_valid_mask.sum()),
        'binding_topic': min(topic_coverage, key=topic_coverage.get) if topic_coverage else None,
        'topics': {},
    }
    for ckey in sorted(cvt_info.keys()):
        cinfo = cvt_info[ckey]
        topic = cinfo['topic']
        info = topic_summary[topic]
        conversion_stats['topics'][topic] = {
            'msg_count': topic_msgcount.get(topic, 0),
            'coverage': topic_coverage.get(topic),
            'frames_written': info['frames_written'],
            'interp_samples': interp_counts.get(topic),
            'status': info['status'],
        }
    with open(os.path.join(args.dst_dir, 'conversion_stats.json'), 'w') as f:
        json.dump(conversion_stats, f, indent=2)

    if not args.no_render_video:
        from tartandriver_utils.video_utils import (
            render_dataset_videos, collect_video_hud_odom, collect_video_hud_imu, load_video_config)
        print(f'\n[stage] +{timedelta(seconds=int(time.time() - full_start))} 6. Rendering videos...')
        viz_dir = os.path.join(args.dst_dir, 'viz')

        vcfg = load_video_config(args.video_config)
        hud = vcfg["hud"]

        hud_data = None
        imu_data = None
        if hud is not None:
            print(f"  [hud] collecting odometry from {hud['odom_kitti_dir']}")
            hud_data = collect_video_hud_odom(args.dst_dir, hud['odom_kitti_dir'])
            print(f"  [hud] collecting imu data from {hud['imu_kitti_dir']}")
            imu_data = collect_video_hud_imu(args.dst_dir, hud['imu_kitti_dir'])

        modality_dirs = {f"{c['group']}/{c['name']}": c['dir'] for c in filt_cvt_info.values()}
        render_dataset_videos(modality_dirs, vcfg, viz_dir, hud_data=hud_data, imu_data=imu_data)

    print(f'\nDone processing {queue["target_times"].shape[0]} frames.')
