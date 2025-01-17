import os
import yaml
import argparse

import numpy as np

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tartandriver_utils.ros_utils import stamp_to_time

from ros_torch_converter.converter import str_to_cvt_class

"""
Script to create kitti-formatted datasets from ros2 bags
General algo is something like this:
    1. First do a pass through to figure out all the target times
    2. Then do another pass through to actually convert messages, etc
"""

def setup_queue(reader, config):
    """
    Initialize message queues based on config
    """
    start_time = reader.start_time * 1e-9
    end_time = reader.end_time * 1e-9

    target_times = np.arange(start_time, end_time, config['dt'])

    queue = {
        'target_times': target_times,
        'topic_times': {},
        'topic_error': {},
    }

    for topic_data in config['data']:
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
    parser.add_argument('--src_dir', type=str, required=True, help='path to input dir')
    parser.add_argument('--dst_dir', type=str, required=True, help='path to output dir')
    parser.add_argument('--dryrun', action='store_true', help='set this flag to check data w/o parsing it')
    parser.add_argument('--use_bag_time', action='store_true', help='set this flag to use bag time for all stamps (not recommended)')
    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        x = input('{} exists. Overwrite? [Y/n]'.format(args.dst_dir))
        if x == 'n':
            exit(0)

    os.makedirs(args.dst_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, 'r'))
    target_topics = [x['topic'] for x in config['data']]
    topic_to_msgtype = {x['topic']:x['type'] for x in config['data']}
    topic_to_name = {x['topic']:x['name'] for x in config['data']}

    bag_fps = sorted([x for x in os.listdir(args.src_dir) if '.mcap' in x])

    print('processing these bags:')
    for bfp in bag_fps:
        print('\t' + bfp)

    bagpath = Path(args.src_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in target_topics]

        assert check_connections(connections, target_topics), "missing topics"

        queue = setup_queue(reader, config)

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if hasattr(msg, "header") and not args.use_bag_time:
                msg_time = stamp_to_time(msg.header.stamp)
            else:
                msg_time = timestamp * 1e-9

            tdiffs = queue['target_times'] - msg_time

            better_mask = np.abs(tdiffs) < queue['topic_error'][topic]

            if not config['backward_interpolation']:
                better_mask = better_mask & (tdiffs >= 0.)

            queue['topic_error'][topic][better_mask] = np.abs(tdiffs)[better_mask]
            queue['topic_times'][topic][better_mask] = msg_time

    
    # debug code
    if args.dryrun:
        import matplotlib.pyplot as plt
        plt.plot(queue['target_times'], marker='.', label='target_times')

        for topic in target_topics:
            times = queue['topic_times'][topic]
            error = queue['topic_error'][topic]
            x = np.arange(len(times))
            mask = error < config['interp_tol']
            plt.plot(x[mask], queue['topic_times'][topic][mask], marker='.', label="{} ({} bad)".format(topic, len(mask) - mask.sum()))

        plt.title('time sync graph')
        plt.legend()
        plt.show()
        exit(0)

    ##  do some proc to get consecutive segments
    all_valid_mask = np.ones(len(queue['target_times']), dtype=bool)
    for topic, err in queue['topic_error'].items():
        all_valid_mask = all_valid_mask & (err < config['interp_tol'])

    queue['target_times'] = queue['target_times'][all_valid_mask]

    for topic in queue['topic_times'].keys():
        queue['topic_times'][topic] = queue['topic_times'][topic][all_valid_mask]
        queue['topic_error'][topic] = queue['topic_error'][topic][all_valid_mask]

    print('keeping {}/{} potential frames.'.format(all_valid_mask.sum(), all_valid_mask.shape[0]))

    np.savetxt(os.path.join(args.dst_dir, 'target_timestamps.txt'), queue['target_times'])

    ## setup folder structure/populate timestamps
    for topic_config in config['data']:
        topic = topic_config['topic']
        topic_dir = os.path.join(args.dst_dir, topic_config['name'])
        os.makedirs(topic_dir, exist_ok=True)

        np.savetxt(os.path.join(topic_dir, 'timestamps.txt'), queue['topic_times'][topic])
        np.savetxt(os.path.join(topic_dir, 'errors.txt'), queue['topic_error'][topic])

    checks = {k:[] for k in target_topics}

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
                print('topic {} msg for frames {}'.format(topic, idxs))
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

