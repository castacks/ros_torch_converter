import os
import yaml
import argparse
import copy
import time
from datetime import timedelta
from multiprocessing import Pool, cpu_count, Manager, set_start_method, Queue
from threading import Thread
import sys

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.os_utils import load_yaml

from ros_torch_converter.converter import str_to_cvt_class
from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.datatypes.intrinsics import CameraInfoTorch

"""
Script to create kitti-formatted datasets from ros2 bags
General algo is something like this:
    1. First do a pass through to figure out all the target times
    2. Then do another pass through to actually convert messages, etc
"""

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

def get_filtered_config(connections, config):
    """
    Check all required topics in the config are present in connections.
    
    Returns:
        filtered_config: in format {'topic': {kwargs}, ...}.
    """
    connection_topics = [x.topic for x in connections]
    missing_topics = []
    filtered_config = {}
    for topic_cfg in config['topics']:
        topic = topic_cfg['topic']
        if topic not in connection_topics:
            if not topic_cfg.get('optional'):
                missing_topics.append(topic)
        else:
            filtered_config[topic] = copy.deepcopy(topic_cfg)
            del filtered_config[topic]['topic'] # redundant
    
    assert not missing_topics, "Bag missing config required topics: {}".format(missing_topics)

    return filtered_config

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

def display_progress_monitor(progress_queue, topic_list, bag, use_color=False):
    """
    Monitor and display progress in a live-updating dashboard style.
    """
    import sys
    
    # Color codes (skip reds as requested)
    COLORS = [
        '\033[92m',  # Green
        '\033[94m',  # Blue
        '\033[96m',  # Cyan
        '\033[93m',  # Yellow
        '\033[95m',  # Magenta
        '\033[34m',  # Dark Blue
        '\033[36m',  # Dark Cyan
        '\033[32m',  # Dark Green
        '\033[35m',  # Dark Magenta
    ]
    RESET = '\033[0m'
    GREEN = '\033[92m'
    GRAY = '\033[90m'
    
    # Initialize status for all topics
    topic_status = {topic: {'status': 'waiting', 'progress': 0, 'total': 0, 'fps': 0.0} for topic in topic_list}
    topic_colors = {topic: COLORS[i % len(COLORS)] for i, topic in enumerate(topic_list)}
    
    def print_dashboard():
        # Clear screen and move cursor to top
        sys.stdout.write('\033[2J\033[H')
        
        print("=" * 100)
        print(f"bag: {bag}")
        print(f"{'Topic':<50} {'Status':<15} {'Progress':<25} {'Speed':>10}")
        print("=" * 100)
        
        for topic in topic_list:
            status = topic_status[topic]
            color = topic_colors[topic] if use_color else ''
            
            # Format status - always use 15 character width for visible text
            if status['status'] == 'completed':
                if use_color:
                    # "✓ COMPLETE" is 10 chars, need 5 more spaces, then wrap in color
                    status_str = f"{GREEN}✓ COMPLETE     {RESET}"
                else:
                    status_str = f"{'✓ COMPLETE':<15}"
                progress_str = f"{status['progress']}/{status['total']}"
                bar = "█" * 20
                fps_str = f"{status['fps']:>6.1f} fps"
            elif status['status'] == 'error':
                status_str = f"{'✗ ERROR':<15}"
                progress_str = "0/0"
                bar = " " * 20
                fps_str = "  0.0 fps"
            elif status['status'] == 'processing':
                status_str = f"{'Processing':<15}"
                progress_str = f"{status['progress']}/{status['total']}"
                pct = status['progress'] / status['total'] if status['total'] > 0 else 0
                filled = int(20 * pct)
                bar = "█" * filled + "░" * (20 - filled)
                fps_str = f"{status['fps']:>6.1f} fps"
            else:  # waiting/starting
                if use_color:
                    # "Waiting" is 7 chars, need 8 more spaces
                    status_str = f"{GRAY}Waiting        {RESET}"
                else:
                    status_str = f"{'Waiting':<15}"
                progress_str = "0/0"
                bar = "░" * 20
                fps_str = "  0.0 fps"
            
            topic_display = f"{color}{topic:<50}{RESET}" if use_color else f"{topic:<50}"
            print(f"{topic_display} {status_str} [{bar}] {progress_str:>10} {fps_str:>10}")
        
        print("=" * 100)
        sys.stdout.flush()
    
    # Print initial dashboard
    print_dashboard()
    
    # Update dashboard as progress comes in
    while True:
        try:
            # Non-blocking check with timeout
            try:
                update = progress_queue.get(timeout=0.5)
                topic, status, progress, total, fps = update
                topic_status[topic] = {'status': status, 'progress': progress, 'total': total, 'fps': fps}
                print_dashboard()
            except:
                # Timeout - check if all completed
                all_done = all(s['status'] in ['completed', 'error'] for s in topic_status.values())
                if all_done:
                    break
        except KeyboardInterrupt:
            break
    
    # Final display
    print_dashboard()

def process_topic_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing that processes a single topic.
    Args is a tuple of (bagpath, topic, topic_config_item, args, topic_times, camera_info_torch, n_frames, color_code, progress_queue)
    """
    bagpath, topic, topic_config_item, args, topic_times, camera_info_torch, n_frames, color_code, progress_queue = args_tuple
    
    # Send initial status
    progress_queue.put((topic, 'starting', 0, 0, 0.0))
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    checks = []
    
    try:
        with AnyReader([bagpath], default_typestore=typestore) as reader:
            # Find the matching connection in this reader instance
            matching_connections = [x for x in reader.connections if x.topic == topic]
            
            if not matching_connections:
                progress_queue.put((topic, 'error', 0, 0, 0.0))
                return topic, checks
            
            start = time.time()
            last_idx = -1
            msg_count = 0
            processed_count = 0
            
            for conn, timestamp, rawdata in reader.messages(connections=matching_connections):
                msg_count += 1
                
                try:
                    msg = reader.deserialize(rawdata, conn.msgtype)
                except Exception as e:
                    continue
                
                msg_time = timestamp * 1e-9
                if hasattr(msg, "header") and not args.use_bag_time:
                    stamp_time = stamp_to_time(msg.header.stamp)
                    if stamp_time != 0: # assume bag time if unpopulated stamp
                        msg_time = stamp_time

                target_diffs = np.abs(topic_times - msg_time)
                idxs = np.argwhere(target_diffs < 1e-8).flatten()

                if len(idxs) > 0:
                    last_idx = max(last_idx, idxs[0].item())
                    processed_count += 1
                    dur = time.time()-start
                    rate = last_idx/dur if dur > 0 else 0
                    
                    # Send progress update every 50 frames
                    if processed_count % 50 == 0 or processed_count == n_frames:
                        progress_queue.put((topic, 'processing', processed_count, n_frames, rate))
                    
                    checks.append(idxs)

                    torch_dtype = str_to_cvt_class[topic_config_item['type']]
                    name = topic_config_item['name']

                    try:
                        # Convert message with optional rectification
                        if camera_info_torch is not None:
                            torch_data = torch_dtype.from_rosmsg(msg, camera_info_torch=camera_info_torch, rectify=True)
                        else:
                            torch_data = torch_dtype.from_rosmsg(msg)

                        base_dir = os.path.join(args.dst_dir, name)
                        for idx in idxs:
                            if torch_data.stamp == -1:
                                torch_data.stamp = topic_times[idx]
                            torch_data.to_kitti(base_dir, idx)
                            
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        continue
        
        dur = time.time() - start
        rate = n_frames/dur if dur > 0 else 0
        progress_queue.put((topic, 'completed', n_frames, n_frames, rate))
        
        return topic, checks
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_queue.put((topic, 'error', 0, n_frames, 0.0))
        return topic, []

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
    parser.add_argument('--skip_tf', action='store_true', help='set this flag to skip TF processing (useful if TF tree is broken)')
    parser.add_argument('--rectify', action='store_true', help='set this flag to rectify compressed images using camera_info (requires camera_info topics in bag)')
    parser.add_argument('--num_workers', type=str, default=None, help='number of parallel workers (default: min of topics and CPU cores, or "max" to use all CPU cores)')
    parser.add_argument('--color', action='store_true', help='use colored output for different topics')
    args = parser.parse_args()

    if os.path.exists(args.dst_dir) and not args.force:
        x = input('{} exists. Overwrite? [Y/n]'.format(args.dst_dir))
        if x == 'n':
            exit(0)

    config = load_yaml(args.config)
    dt = config['dt']
    all_topics = [x['topic'] for x in config['topics']]

    # Check for missing message type converters upfront
    check_missing_types(config)

    bag_fps = sorted([x for x in os.listdir(args.src_dir) if '.mcap' in x])

    print('processing these bags:')
    for bfp in bag_fps:
        print('\t' + bfp)

    bagpath = Path(args.src_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    frame_list = set()

    print('checking timestamps...')
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        # Do not add topics with 0 count to the queue, else sync issues
        connections = [x for x in reader.connections if x.msgcount > 0 and x.topic in all_topics]

        # update target topics to be only valid topics
        topic_config = get_filtered_config(connections, config)
        target_topics = [k for k in topic_config]

        queue = setup_queue(reader, target_topics, dt)

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            msg_time = timestamp * 1e-9
            if hasattr(msg, "header") and not args.use_bag_time:
                stamp_time = stamp_to_time(msg.header.stamp)
                if stamp_time != 0: # assume bag time if unpopulated stamp
                    msg_time = stamp_time
                frame_list.add(msg.header.frame_id)

            if hasattr(msg, "child_frame_id"):
                frame_list.add(msg.child_frame_id)

            tdiffs = queue['target_times'] - msg_time

            better_mask = np.abs(tdiffs) < queue['topic_error'][topic]

            if not config['backward_interpolation']:
                better_mask = better_mask & (tdiffs >= 0.)

            queue['topic_error'][topic][better_mask] = np.abs(tdiffs)[better_mask]
            queue['topic_times'][topic][better_mask] = msg_time

    #update the tf tree
    frame_list = list(frame_list)

    # temp broken rtk transform fix
    temp_ignore = ['gq7_imu_link', 'earth']
    for fr in temp_ignore:
        if fr in frame_list:
            frame_list.remove(fr)

    has_calib_file = False
    if 'calib_file' in config.keys():
        print('applying calib file from config...')
        calib_config = yaml.safe_load(open(config['calib_file'], 'r'))
        has_calib_file = True

    elif args.calib_file is not None:
        print('applying calib file from cli...')
        calib_config = yaml.safe_load(open(args.calib_file, 'r'))
        has_calib_file = True

    if not has_calib_file:
        print('no calib file provided. Note that for Yamaha data this is probably wrong!')

    if args.skip_tf:
        print('Skipping TF processing as requested...')
        tf_manager = None
        tf_tmin = -np.inf
        tf_tmax = np.inf
    else:
        print('handling tf...')
        tf_manager = TfManager.from_rosbag(bagpath, device='cuda')

        if has_calib_file:
            tf_manager.update_from_calib_config(calib_config)

        tf_tmin, tf_tmax = tf_manager.get_valid_times_from_list(frame_list)

    ##  do some proc to get consecutive segments
    all_valid_mask = np.ones(len(queue['target_times']), dtype=bool)

    for topic, err in queue['topic_error'].items():
        all_valid_mask = all_valid_mask & (err < config['interp_tol'])

    for topic, times in queue['topic_times'].items():
        all_valid_mask = all_valid_mask & (times > tf_tmin) & (times < tf_tmax)

    assert all_valid_mask.any(), "Topics not sync'ed!"

    queue['target_times'] = queue['target_times'][all_valid_mask]

    for topic in queue['topic_times'].keys():
        queue['topic_times'][topic] = queue['topic_times'][topic][all_valid_mask]
        queue['topic_error'][topic] = queue['topic_error'][topic][all_valid_mask]

    print('keeping {}/{} potential frames.'.format(all_valid_mask.sum(), all_valid_mask.shape[0]))
    n_frames = all_valid_mask.shape[0]

    os.makedirs(args.dst_dir, exist_ok=True)
    np.savetxt(os.path.join(args.dst_dir, 'target_timestamps.txt'), queue['target_times'])

    ## setup folder structure/populate timestamps
    for topic, cfg in topic_config.items():
        topic_dir = os.path.join(args.dst_dir, cfg['name'])
        os.makedirs(topic_dir, exist_ok=True)

        np.savetxt(os.path.join(topic_dir, 'timestamps.txt'), queue['topic_times'][topic])
        np.savetxt(os.path.join(topic_dir, 'errors.txt'), queue['topic_error'][topic])

    checks = {k:[] for k in target_topics}

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
    
    # note that behavior is non-deterministic if a topic has multiple msgs with the same timestamp
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        # If rectification is requested, collect camera_info messages
        camera_info_cache = {}
        if args.rectify:
            # Cache for camera_info messages
            camera_info_cache = {}
            camera_info_topics = [topic for topic in target_topics if 'camera_info' in topic]
            if camera_info_topics:
                print(f"Caching camera_info from {len(camera_info_topics)} topics for rectification...")
                camera_info_connections = [x for x in reader.connections if x.topic in camera_info_topics]
                for connection, timestamp, rawdata in reader.messages(connections=camera_info_connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    # Convert to CameraInfoTorch for rectification
                    camera_info_torch = CameraInfoTorch.from_rosmsg(msg, device='cpu')
                    camera_info_cache[connection.topic] = camera_info_torch
                print(f"Cached {len(camera_info_cache)} camera_info messages")
            else:
                print("WARNING: --rectify flag set but no camera_info topics found in bag!")
        
        connections = [x for x in reader.connections if x.topic in target_topics]

    # Process topics in parallel using multiprocessing
    start = time.time()
    
    # Define color palette (ANSI color codes) - skip reds
    COLORS = [
        '\033[92m',  # Green
        '\033[94m',  # Blue
        '\033[96m',  # Cyan
        '\033[93m',  # Yellow
        '\033[95m',  # Magenta
        '\033[32m',  # Dark Green
        '\033[34m',  # Dark Blue
        '\033[36m',  # Dark Cyan
        '\033[35m',  # Dark Magenta
    ]
    
    # Create a progress queue for live updates
    progress_queue = Manager().Queue()
    
    # Prepare arguments for each topic - pass minimal data to avoid serialization issues
    process_args = []
    for idx, con in enumerate(connections):
        topic = con.topic
        topic_config_item = topic_config[topic]
        topic_times = queue['topic_times'][topic]
        
        # Get camera_info for this specific topic if needed
        camera_info_torch = None
        if args.rectify and 'CompressedImage' in topic_config_item['type']:
            base_topic = topic.replace('/image_raw/compressed', '').replace('/compressed', '')
            camera_info_topic = base_topic + '/camera_info'
            camera_info_torch = camera_info_cache.get(camera_info_topic, None)
        
        # Assign a color to each topic (cycle through colors if more topics than colors)
        color_code = COLORS[idx % len(COLORS)]
        
        process_args.append((bagpath, topic, topic_config_item, args, topic_times, camera_info_torch, n_frames, color_code, progress_queue))
    
    # Determine number of workers
    if args.num_workers is None:
        num_workers = min(len(connections), cpu_count())
    elif args.num_workers == "max":
        num_workers = cpu_count()
    else:
        num_workers = int(args.num_workers)
    
    print(f"\nProcessing {len(connections)} topics in parallel using {num_workers} workers...\n")
    
    # Start the progress monitor in a separate thread
    monitor_thread = Thread(target=display_progress_monitor, args=(progress_queue, [con.topic for con in connections], args.src_dir, args.color))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    with Pool(processes=num_workers) as pool:
        try:
            # Use a timeout to avoid hanging indefinitely
            async_result = pool.map_async(process_topic_wrapper, process_args)
            
            # Wait for results with timeout
            results = async_result.get(timeout=7200)  # 2 hour timeout
        except TimeoutError:
            print("\nERROR: Processing timed out after 2 hours")
            pool.terminate()
            pool.join()
            raise
    
    # Wait for monitor to finish displaying
    monitor_thread.join(timeout=2)
    
    # Collect results
    checks = {}
    for topic, topic_checks in results:
        checks[topic] = topic_checks
    
    print(f"\n{'='*100}")
    print(f"All {len(connections)} topics completed successfully!")
    print(f"{'='*100}")

    dur = time.time()-start
    rate = n_frames/dur if dur > 0 else 0
    print(f'\nTotal processing time: {timedelta(seconds=int(dur))}')
    print(f'Overall rate: {rate:.1f} frames/sec')
    print(f"{'='*60}\n")

    ## check that all idxs got filled
    checks = {k:np.sort(np.concatenate(v)) if len(v) > 0 else np.array([]) for k,v in checks.items()}

    print('Verification:')
    print("{}/{} valid frames for dataset".format(all_valid_mask.sum(), all_valid_mask.shape[0]))
    for topic, idxs in checks.items():
        if len(idxs) > 0:
            valid = all(np.unique(idxs) == np.arange(all_valid_mask.sum()))
            status = "✓" if valid else "✗"
            print(f'  {status} {topic}: {len(np.unique(idxs))}/{all_valid_mask.sum()} frames')
        else:
            print(f'  ✗ {topic}: 0/{all_valid_mask.sum()} frames (NO DATA)')
    
    print(f'\nDone processing {queue["target_times"].shape[0]} frames.')
