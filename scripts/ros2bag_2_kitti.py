import os
import tqdm
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tabulate import tabulate

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.os_utils import load_yaml

from ros_torch_converter.converter import str_to_cvt_class
from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.utils import select_ros1_bags
from ros_torch_converter.datatypes.base import TimeSpec
from ros_torch_converter.datatypes.intrinsics import CameraInfoTorch

"""
Script to create kitti-formatted datasets from ros2 bags
General algo is something like this:
    1. First do a pass through to figure out all the target times
    2. Then do another pass through to actually convert messages, etc
"""

def setup_queue(reader, config, target_times_override=None, window=None):
    """
    Initialize message queues based on config.

    target_times_override: if given (a 1D array of timestamps, e.g. an existing extraction's
    target_timestamps.txt), use it as the FIXED time grid instead of arange(start,end,dt). This is
    how `--align_to` additive extraction lands new modalities on an existing tree's grid.

    window: optional (t_start, t_end) in seconds. Clamps the arange grid to the window, or filters the
    override grid to it, so extraction is scoped to a time window (pairs with the reader's start/stop).
    """
    if target_times_override is not None:
        target_times = np.asarray(target_times_override, dtype=float).reshape(-1)
        if window is not None:
            target_times = target_times[(target_times >= window[0]) & (target_times <= window[1])]
    else:
        start_time = reader.start_time * 1e-9
        end_time = reader.end_time * 1e-9
        if window is not None:
            start_time = max(start_time, window[0])
            end_time = min(end_time, window[1])
        target_times = np.arange(start_time, end_time, config['dt'])

    queue = {
        'target_times': target_times,
        'topic_times': {},
        'topic_error': {},
    }

    for topic_data in config['topics']:
        queue['topic_times'][topic_data['topic']] = -np.ones(len(target_times))
        queue['topic_error'][topic_data['topic']] = float('inf') * np.ones(len(target_times))

    return queue

def check_connections(connections, target_topics):
    """
    Check that all topics in target_topics in connections
    """
    valid = True
    connection_topics = [x.topic for x in connections]
    for topic in target_topics:
        if topic not in connection_topics:
            print('bag missing config topic {}!'.format(topic))
            valid = False

    return valid

if __name__ == '__main__':
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
    parser.add_argument('--ros1', action='store_true', help='read ROS1 .bag files (merges all per-sensor .bag files in src_dir) instead of ROS2 mcap. Default: autodetect by extension.')
    parser.add_argument('--align_to', type=str, default=None,
                        help='ADDITIVE extraction: path to an existing KITTI tree whose '
                             'target_timestamps.txt is reused as the FIXED time grid. Only the '
                             'configured (new) topics are extracted, index-aligned 1:1 with that tree; '
                             'the cross-topic sync filter, target_timestamps.txt, tf/ and sync_plot are '
                             'NOT recomputed/written (existing groups are untouched). Use to add a '
                             'modality later without re-extracting or re-syncing what is already there.')
    parser.add_argument('--time-window', dest='time_window', type=float, nargs=2, default=None,
                        metavar=('T_START', 'T_END'),
                        help='only read bag messages within [T_START, T_END] (seconds, same epoch as the '
                             'data timestamps). Pushed down into the rosbags reader as a time-bounded '
                             '(index-based) read, so the whole bag is never scanned/decoded — useful for '
                             'extracting a handful of frames from a large bag. Also clamps the sync grid '
                             '(and, with --align_to, filters the reused grid) to the window.')
    args = parser.parse_args()

    # ns bounds for time-bounded rosbags reads (None => unbounded). rosbags reader start/stop act on the
    # bag record time; the sync grid clamp below uses the same seconds window on the data-timestamp clock.
    win_start_ns = int(args.time_window[0] * 1e9) if args.time_window else None
    win_stop_ns = int(args.time_window[1] * 1e9) if args.time_window else None

    if os.path.exists(args.dst_dir) and not args.force:
        x = input('{} exists. Overwrite? [Y/n]'.format(args.dst_dir))
        if x == 'n':
            exit(0)

    config = load_yaml(args.config)

    cvt_info = {}
    for tconf in config['topics']:
        name = f"{tconf['group']}/{tconf['name']}"
        assert name not in cvt_info.keys()

        cvt_info[name] = {
            'name': tconf['name'],
            'group': tconf['group'],
            'topic': tconf['topic'],
            'msgtype': tconf['type'],
            # per-topic rectify override (default True); lets e.g. RGB be rectified while
            # thermal is kept raw in the same run when global --rectify is set.
            'rectify': tconf.get('rectify', True),
            'dir': os.path.join(args.dst_dir, tconf['group'], tconf['name'])
        }

    target_topics = set([x['topic'] for x in cvt_info.values()])

    # Check for missing message type converters upfront
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

    # Autodetect ROS1 if not explicitly set: ROS1 src_dirs hold .bag files,
    # ROS2 src_dirs are rosbag2 dirs containing .mcap.
    ros1 = args.ros1
    if not ros1:
        has_bag = any(x.endswith('.bag') for x in os.listdir(args.src_dir))
        has_mcap = any('.mcap' in x for x in os.listdir(args.src_dir))
        if has_bag and not has_mcap:
            ros1 = True
            print('autodetected ROS1 .bag files in src_dir')

    # ROS1: --src_dir may point at a whole bag folder. Pre-filter with a cheap topic peek so we
    # only open/index the bags that actually carry a configured topic (select_ros1_bags reads
    # just each bag's connection records, not its message index). ROS2: pass the rosbag2 dir.
    if ros1:
        bag_paths = select_ros1_bags(args.src_dir, target_topics)
    else:
        bag_paths = [Path(args.src_dir)]

    print('processing these bags:')
    for bp in bag_paths:
        print('\t' + bp.name)

    typestore = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)

    frame_list = set()

    ## initial simple implementation of interp topics
    topics_to_interp = [_ci['topic'] for _ci in cvt_info.values() if str_to_cvt_class[_ci['msgtype']].time_spec == TimeSpec.INTERP]
    interp_buf = {k:[] for k in topics_to_interp}

    print('collecting full interp data for the following topics:')
    for t in topics_to_interp:
        print(f'\t{t}')

    ##print proc check table
    tabdata = [['Name', 'Group', 'Msg Type', 'Topic', 'Save Path']]
    for _ci in cvt_info.values():
        tabdata.append([_ci[k] for k in ['name', 'group', 'msgtype', 'topic', 'dir']])

    print(tabulate(tabdata, headers='firstrow', tablefmt='outline'))

    if not args.force:
        x = input('does this look correct?[Y/n]')
        if x == 'n':
            exit(0)

    print('checking timestamps...')
    with AnyReader(bag_paths, default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in target_topics]

        assert check_connections(connections, target_topics), "missing topics"

        target_times_override = None
        if args.align_to:
            _grid_fp = os.path.join(args.align_to, 'target_timestamps.txt')
            target_times_override = np.loadtxt(_grid_fp)
            print('align_to: reusing {} target times from {}'.format(
                len(np.atleast_1d(target_times_override)), _grid_fp))
        queue = setup_queue(reader, config, target_times_override=target_times_override,
                             window=args.time_window)

        for connection, timestamp, rawdata in reader.messages(connections=connections,
                                                              start=win_start_ns, stop=win_stop_ns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if hasattr(msg, "header") and not args.use_bag_time:
                msg_time = stamp_to_time(msg.header.stamp)
                frame_list.add(msg.header.frame_id)
            else:
                msg_time = timestamp * 1e-9

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

    if args.skip_tf or args.align_to:
        print('Skipping TF processing{}...'.format(' (align_to: reuse existing tree tf)' if args.align_to else ' as requested'))
        tf_manager = None
        tf_tmin = -np.inf
        tf_tmax = np.inf
    else:
        print('handling tf...')
        # auto-detect device so this runs on CPU-only boxes (parv-dev-cpu) as well as GPU boxes
        import torch as _torch
        _tf_device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        tf_manager = TfManager.from_rosbag(args.src_dir, device=_tf_device, ros1=ros1)

        if has_calib_file:
            tf_manager.update_from_calib_config(calib_config)

        tf_tmin, tf_tmax = tf_manager.get_valid_times_from_list(frame_list)

    ##  do some proc to get consecutive segments
    all_valid_mask = np.ones(len(queue['target_times']), dtype=bool)

    if args.align_to:
        # Additive mode: keep the existing fixed grid as-is (no cross-topic filtering / re-indexing).
        # Grid points where the new modality has no sample within interp_tol stay -1 in topic_times
        # and simply get no frame written (a gap at that index), preserving 1:1 index alignment with
        # the existing tree. Do NOT (re)write target_timestamps.txt (don't clobber the existing tree).
        print('align_to: keeping all {} fixed grid points; no cross-topic sync filtering.'.format(
            len(queue['target_times'])))
    else:
        for topic, err in queue['topic_error'].items():
            all_valid_mask = all_valid_mask & (err < config['interp_tol'])

        for topic, times in queue['topic_times'].items():
            all_valid_mask = all_valid_mask & (times > tf_tmin) & (times < tf_tmax)

        assert all_valid_mask.any(), "topics not sync'ed!"

        queue['target_times'] = queue['target_times'][all_valid_mask]

        for topic in queue['topic_times'].keys():
            queue['topic_times'][topic] = queue['topic_times'][topic][all_valid_mask]
            queue['topic_error'][topic] = queue['topic_error'][topic][all_valid_mask]

        print('keeping {}/{} potential frames.'.format(all_valid_mask.sum(), all_valid_mask.shape[0]))

    n_frames = all_valid_mask.shape[0]

    os.makedirs(args.dst_dir, exist_ok=True)
    if not args.align_to:
        np.savetxt(os.path.join(args.dst_dir, 'target_timestamps.txt'), queue['target_times'])

    ## setup folder structure/populate timestamps
    for cvt_config in cvt_info.values():
        topic = cvt_config['topic']
        topic_dir = os.path.join(args.dst_dir, cvt_config['dir'])
        os.makedirs(topic_dir, exist_ok=True)

        np.savetxt(os.path.join(topic_dir, 'timestamps.txt'), queue['topic_times'][topic])
        np.savetxt(os.path.join(topic_dir, 'errors.txt'), queue['topic_error'][topic])

    checks = {k:[] for k in target_topics}

    if tf_manager is not None:
        tf_manager.to_kitti(args.dst_dir)

        print('TF TREE:\n')
        print(tf_manager.tf_tree)

    # Additive mode reuses the existing tree's grid -> don't (re)write the sync plot for it.
    if not args.align_to:
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
    # import sys
    pbars = {k:tqdm.tqdm(desc=k, total=all_valid_mask.sum(), position=i) for i,k in enumerate(cvt_info.keys())}
    interp_buf = {k:[] for k in topics_to_interp}

    with AnyReader(bag_paths, default_typestore=typestore) as reader:
        # If rectification is requested, collect camera_info messages
        if args.rectify:
            # Cache for camera_info messages
            camera_info_cache = {}
            camera_info_topics = [topic for topic in target_topics if 'camera_info' in topic]
            if camera_info_topics:
                print(f"Caching camera_info from {len(camera_info_topics)} topics for rectification...")
                camera_info_connections = [x for x in reader.connections if x.topic in camera_info_topics]
                for connection, timestamp, rawdata in reader.messages(connections=camera_info_connections,
                                                                      start=win_start_ns, stop=win_stop_ns):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    # Convert to CameraInfoTorch for rectification
                    camera_info_torch = CameraInfoTorch.from_rosmsg(msg, device='cpu')
                    camera_info_cache[connection.topic] = camera_info_torch
                print(f"Cached {len(camera_info_cache)} camera_info messages")
            else:
                print("WARNING: --rectify flag set but no camera_info topics found in bag!")
        
        connections = [x for x in reader.connections if x.topic in target_topics]

        assert check_connections(connections, target_topics), "missing topics"

        # camera_info topics whose sibling image is rectified at extraction -> their stored
        # calibration must use the rectified convention (D=0, K=P[:3,:3], R=I) so the metadata
        # matches the undistorted pixels (mirrors the image-rectify gate below).
        _RECTIFIABLE = ('CompressedImage', 'Thermal16bitCompressedImage')
        rectified_info_topics = set()
        if args.rectify:
            for _e in cvt_info.values():
                if _e['msgtype'] in _RECTIFIABLE and _e.get('rectify', True):
                    _base = _e['topic'].replace('/image_raw/compressed', '').replace('/image/compressed', '').replace('/compressed', '')
                    rectified_info_topics.add(_base + '/camera_info')

        for connection, timestamp, rawdata in reader.messages(connections=connections,
                                                              start=win_start_ns, stop=win_stop_ns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic
            cinfo_to_update = [(k,v) for k,v in cvt_info.items() if v['topic'] == topic]

            for _ckey, _cinfo in cinfo_to_update:
                base_dir = _cinfo['dir']
                torch_dtype = str_to_cvt_class[_cinfo['msgtype']]

                if hasattr(msg, "header") and not args.use_bag_time:
                    msg_time = stamp_to_time(msg.header.stamp)
                else:
                    msg_time = timestamp * 1e-9
                    
                ## handle interp data
                if topic in topics_to_interp:
                    torch_data = torch_dtype.from_rosmsg(msg)
                    torch_data.stamp = msg_time

                    if (len(interp_buf[topic]) == 0) or (msg_time - interp_buf[topic][-1].stamp > 1e-16):
                        interp_buf[topic].append(torch_data)

                ## handle sync data
                target_diffs = np.abs(queue['topic_times'][topic] - msg_time)
                idxs = np.argwhere(target_diffs < 1e-16).flatten()

                if len(idxs) > 0:
                    checks[topic].append(idxs)

                    torch_data = torch_dtype.from_rosmsg(msg)
                    
                    camera_info_torch = None
                    # Rectify this image only if the global flag is set, the per-topic override is
                    # not disabled, and it's a compressed image type with a camera_info.
                    if args.rectify and _cinfo.get('rectify', True) and _cinfo['msgtype'] in ('CompressedImage', 'Thermal16bitCompressedImage'):
                        # Try to find a matching camera_info topic
                        # Assume camera_info topic is same base topic with /camera_info suffix
                        base_topic = topic.replace('/image_raw/compressed', '').replace('/image/compressed', '').replace('/compressed', '')
                        camera_info_topic = base_topic + '/camera_info'
                        camera_info_torch = camera_info_cache.get(camera_info_topic, None)
                        if camera_info_torch is None:
                            print(f"\nWARNING: No camera_info found for {topic}, skipping rectification")
                    
                    # Convert message with optional rectification
                    if camera_info_torch is not None:
                        torch_data = torch_dtype.from_rosmsg(msg, camera_info_torch=camera_info_torch, rectify=True)
                    else:
                        torch_data = torch_dtype.from_rosmsg(msg)

                    # If this is a camera_info whose sibling image was rectified, store it in the
                    # rectified convention so the calibration matches the undistorted pixels.
                    if _cinfo['msgtype'] == 'CameraInfo' and topic in rectified_info_topics:
                        torch_data = torch_data.as_rectified()

                    for idx in idxs:
                        torch_data.to_kitti(base_dir, idx)
                        pbars[_ckey].n = idx
                        pbars[_ckey].refresh()

    for pbar in pbars.values():
        pbar.close()    

    #handle interpolated data
    for cname, cinfo in cvt_info.items():
        if cinfo['topic'] in topics_to_interp:
            cvt_type = str_to_cvt_class[cinfo['msgtype']]
            base_dir = cinfo['dir']
            interp_data = interp_buf[cinfo['topic']]

            print(f"make interp data for dir {base_dir} from topic {cinfo['topic']} ({len(interp_data)} msgs)")
            cvt_type.to_interp(base_dir, interp_data)


    ## check that all idxs got filled (diagnostic only — must not raise; with --align_to + a time window
    ## some grid points legitimately have no sample (a gap), so the filled set is a subset of the grid).
    checks = {k: (np.sort(np.concatenate(v)) if len(v) else np.array([], dtype=int))
              for k, v in checks.items()}

    print('Done processing {} frames.'.format(queue['target_times'].shape[0]))

    for topic, idxs in checks.items():
        uniq = np.unique(idxs)
        expected = np.arange(all_valid_mask.sum())
        valid = (uniq.shape == expected.shape) and bool(np.all(uniq == expected))
        print('{} has all frames: {} ({}/{})'.format(topic, valid, len(uniq), len(expected)), flush=True)