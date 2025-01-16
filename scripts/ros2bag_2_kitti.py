import os
import yaml
import argparse

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


"""
Script to create kitti-formatted datasets from ros2 bags
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config')
    parser.add_argument('--src_dir', type=str, required=True, help='path to input dir')
    parser.add_argument('--dst_dir', type=str, required=True, help='path to output dir')
    args = parser.parse_args()

    bag_fps = sorted([x for x in os.listdir(args.src_dir) if '.mcap' in x])

    print('processing these bags:')
    for bfp in bag_fps:
        print('\t' + bfp)

    bagpath = Path(args.src_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        import pdb;pdb.set_trace()
        connections = reader.connections
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                if connection.topic == '/multisense/left/image_rect_color':
                    import pdb;pdb.set_trace()
                msg = reader.deserialize(rawdata, connection.msgtype)
                print(msg.header.frame_id, msg.header.stamp)
            except:
                print('aaa')
