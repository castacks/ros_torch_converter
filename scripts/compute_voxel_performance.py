import os
import re
import yaml
import argparse

import numpy as np

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from ros_torch_converter.datatypes.pointcloud import PointCloudTorch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='path to input dir')
    args = parser.parse_args()

#    timing_topic = '/lester/vfm_voxels/torch_coordinator/timing'
#    voxel_topic = '/lester/vfm_voxels/dino_voxels_viz'
    timing_topic = '/torch_coordinator/timing'
    voxel_topic = '/dino_voxels_viz'
    target_topics = [timing_topic, voxel_topic]

    num_voxels = []
    vfm_speed = []
    mapping_speed = []
    terrain_speed = []

    bag_fps = sorted([x for x in os.listdir(args.src_dir) if '.mcap' in x])

    print('processing these bags:')
    for bfp in bag_fps:
        print('\t' + bfp)

    bagpath = Path(args.src_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in target_topics]

        i = 0

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            i += 1
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if topic == voxel_topic:
                data = PointCloudTorch.from_rosmsg(msg)
                num_voxels.append(data.pts.shape[0])

            elif topic == timing_topic:
                tokens = msg.data.split('\n')

                for token in tokens:
                    if 'image_featurizer' in token:
                        match = re.search(r"\d+\.\d+", token)
                        if match:
                            num = float(match.group())
                            vfm_speed.append(num)

                    if 'voxel_mapper' in token:
                        match = re.search(r"\d+\.\d+", token)
                        if match:
                            num = float(match.group())
                            mapping_speed.append(num)

                    if 'terrain_estimation' in token:
                        match = re.search(r"\d+\.\d+", token)
                        if match:
                            num = float(match.group())
                            terrain_speed.append(num)

    num_voxels = np.array(num_voxels)
    vfm_speed = np.array(vfm_speed)
    mapping_speed = np.array(mapping_speed)
    terrain_speed = np.array(terrain_speed)

    print('num_voxels  = {:.4f}+-{:.4f}'.format(num_voxels.mean(), num_voxels.std()))
    print('vfm_speed  = {:.4f}+-{:.4f}'.format(vfm_speed.mean(), vfm_speed.std()))
    print('mapping_speed  = {:.4f}+-{:.4f}'.format(mapping_speed.mean(), mapping_speed.std()))
    print('terrain_speed  = {:.4f}+-{:.4f}'.format(terrain_speed.mean(), terrain_speed.std()))
