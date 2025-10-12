import os
import yaml
import argparse
from typing import Sequence, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tartandriver_utils.ros_utils import stamp_to_time

from ros_torch_converter.converter import str_to_cvt_class
from ros_torch_converter.tf_manager import TfManager

import logging

RED = "\033[31m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO)

"""
Script to create kitti-formatted datasets from ros2 bags
General algo is something like this:
    1. First do a pass through to figure out all the target times
    2. Then do another pass through to actually convert messages, etc
"""

def max_dt_subset(times: Sequence[float], dt: float) -> Tuple[List[int], np.ndarray]:
    """
    Pick the largest subset with gaps >= dt, and FORCE-include the first element.
    Assumes `times` are in nondecreasing chronological order.

    Returns:
        indices: list of selected indices into `times`
        values:  numpy array of selected times
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return [], t

    sel = [0]                    # must include the first element
    last = t[0]

    for i in range(1, t.size):
        if t[i] - last >= dt:
            sel.append(i)
            last = t[i]

    return sel, t[sel]


def extract_timestamp(msg,bag_timestamp,use_bag_time):
    if hasattr(msg, "header") and not use_bag_time:
        msg_time = stamp_to_time(msg.header.stamp)
    else:
        msg_time = bag_timestamp * 1e-9
    return msg_time

def setup_queue(reader, config, connections=None):
    """
    Initialize message queues based on config
    """

    start_time = float('inf')
    end_time = -float('inf')
    start_topic = ""
    end_topic = ""

    if "master_topic" in config.keys():
        master_topic = config['master_topic']
        master_conns = [x for x in connections if x.topic == master_topic]
        assert len(master_conns) > 0, "master topic not found in bag!"
        print("Using {} as master topic".format(master_topic))
        connections = master_conns
        master_times = []


    for connection, timestamp, rawdata in reader.messages(connections=connections):

        msg = reader.deserialize(rawdata, connection.msgtype)
        msg_time = extract_timestamp(msg, timestamp, args.use_bag_time)
        if msg_time < start_time:
            start_time = msg_time
            start_topic = connection.topic
        if msg_time > end_time:
            end_time = msg_time 
            end_topic = connection.topic
        
        if "master_topic" in config.keys():
            if connection.topic == master_topic:
                master_times.append(msg_time)
    
    print("Correct start time: ", start_time , "bag start time: ", reader.start_time * 1e-9)
    print("Correct end time: ", end_time , "bag end time: ", reader.end_time * 1e-9)
    print("Start topic: ", start_topic)
    print("End topic: ", end_topic)

    # target_times = np.arange(start_time, end_time, config['dt'])

    target_times = max_dt_subset(master_times, config['dt'])[1]

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
    args = parser.parse_args()

    if os.path.exists(args.dst_dir) and not args.force:
        x = input('{} exists. Overwrite? [Y/n]'.format(args.dst_dir))
        if x == 'n':
            exit(0)

    config = yaml.safe_load(open(args.config, 'r'))
    target_topics = [x['topic'] for x in config['topics']]
    topic_to_msgtype = {x['topic']:x['type'] for x in config['topics']}
    topic_to_name = {x['topic']:x['name'] for x in config['topics']}

    bag_fps = sorted([x for x in os.listdir(args.src_dir) if '.mcap' in x])

    print('processing these bags:')
    for bfp in bag_fps:
        print('\t' + bfp)

    bagpath = Path(args.src_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    frame_list = set()

    print('checking timestamps...')
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in target_topics]

        assert check_connections(connections, target_topics), "missing topics"

        queue = setup_queue(reader, config,connections=connections)

        for connection, timestamp, rawdata in reader.messages(connections=connections):
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
            elif config['backward_interpolation_only']:
                assert config['backward_interpolation'] == True , "set backward_interpolation to True if backward_interpolation_only is True"
                better_mask = better_mask & (tdiffs <= 0.)
            

            queue['topic_error'][topic][better_mask] = np.abs(tdiffs)[better_mask]
            queue['topic_times'][topic][better_mask] = msg_time

    #update the tf tree
    frame_list = list(frame_list)
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

    print('handling tf...')
    # tf_manager = TfManager.from_rosbag(bagpath, device='cuda')
    tf_manager = TfManager(device='cuda')
    if has_calib_file:
        tf_manager.update_from_calib_config(calib_config)

    tf_tmin, tf_tmax = tf_manager.get_valid_times_from_list(frame_list)

    ##  do some proc to get consecutive segments
    all_valid_mask = np.ones(len(queue['target_times']), dtype=bool)

    for topic, err in queue['topic_error'].items():
        all_valid_mask = all_valid_mask & (err < config['interp_tol'])

    for topic, times in queue['topic_times'].items():
        all_valid_mask = all_valid_mask & (times > tf_tmin) & (times < tf_tmax)


    queue['target_times'] = queue['target_times'][all_valid_mask]

    for topic in queue['topic_times'].keys():
        queue['topic_times'][topic] = queue['topic_times'][topic][all_valid_mask]
        queue['topic_error'][topic] = queue['topic_error'][topic][all_valid_mask]

    print('keeping {}/{} potential frames.'.format(all_valid_mask.sum(), all_valid_mask.shape[0]))
    if all_valid_mask.sum() == 0:
        logging.info(f"{RED}No valid frames found! Try increasing interp_tol or adjusting dt{RESET}")

        exit(0)
    # assert all_valid_mask.any(), "topics not sync'ed!"
    n_frames = all_valid_mask.shape[0]

    os.makedirs(args.dst_dir, exist_ok=True)
    np.savetxt(os.path.join(args.dst_dir, 'target_timestamps.txt'), queue['target_times'])

    ## setup folder structure/populate timestamps
    for topic_config in config['topics']:
        topic = topic_config['topic']
        topic_dir = os.path.join(args.dst_dir, topic_config['name'])
        os.makedirs(topic_dir, exist_ok=True)

        np.savetxt(os.path.join(topic_dir, 'timestamps.txt'), queue['topic_times'][topic])
        np.savetxt(os.path.join(topic_dir, 'errors.txt'), queue['topic_error'][topic])

    checks = {k:[] for k in target_topics}

    tf_manager.to_kitti(args.dst_dir)

    print('TF TREE:\n')
    print(tf_manager.tf_tree)

    plt.plot(queue['target_times'], marker='.', label='target_times')

    x = np.arange(len(queue['target_times']))

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
    tf_manager.to_kitti(args.dst_dir)

    if args.dryrun:
        if not args.no_plot:
            plt.show()
        exit(0)

    # note that behavior is non-deterministic if a topic has multiple msgs with the same timestamp
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in target_topics]

        assert check_connections(connections, target_topics), "missing topics"

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic
            name = topic_to_name[topic]

            if hasattr(msg, "header") and not args.use_bag_time:
                msg_time = stamp_to_time(msg.header.stamp)
            else:
                msg_time = timestamp * 1e-9
                
            target_diffs = np.abs(queue['topic_times'][topic] - msg_time)
            idxs = np.argwhere(target_diffs < 1e-8).flatten()

            if len(idxs) > 0:
#                print('topic {} msg for frames {}'.format(topic, idxs))
                print('proc idx {}/{}'.format(idxs[0].item(), n_frames), end='\r')
                checks[topic].append(idxs)

                torch_dtype = str_to_cvt_class[topic_to_msgtype[topic]]
                torch_data = torch_dtype.from_rosmsg(msg)

                base_dir = os.path.join(args.dst_dir, name)
                for idx in idxs:
                    torch_data.to_kitti(base_dir, idx)

    ## check that all idxs got filled
    checks = {k:np.sort(np.concatenate(v)) for k,v in checks.items()}

    print('Done processing {} frames.'.format(queue['target_times'].shape[0]))

    for topic, idxs in checks.items():
        valid = all(np.unique(idxs) == np.arange(all_valid_mask.sum()))
        print('{} has all frames: {}'.format(topic, valid))

