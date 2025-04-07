import argparse
import numpy as np
import open3d as o3d
from mcap.reader import make_reader
from sensor_msgs_py.point_cloud2 import read_points_numpy
import yaml
import os

"""
Script to convert slam point cloud in mcap to pcd.
"""

def main(config):
    all_points = None
    max_points = config['max_points'] if 'max_points' in config else float('inf')
    
    if os.path.exists(config['output_path']):
        print(f"Output path: {config['output_path']} already exists. Exiting.")
        return
    if not os.path.exists(os.path.dirname(config['output_path'])):
        os.makedirs(os.path.dirname(config['output_path']))
    
    print(f"Reading PointCloud2 messages from {config['topic']} in {config['mcap_dir']}...")
    count = 0
    
    with open(config['mcap_dir'], 'rb') as f:
        reader = make_reader(f)
        
        try:
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
            
            pc2_type = get_message('sensor_msgs/msg/PointCloud2')
            
            for schema, channel, message in reader.iter_messages():
                if channel.topic != config['topic']:
                    continue
                    
                if 'sensor_msgs/msg/PointCloud2' in schema.name:
                    ros_msg = deserialize_message(message.data, pc2_type)
                    
                    points = read_points_numpy(ros_msg, field_names=['x', 'y', 'z'], skip_nans=True)
                    
                    if all_points is None:
                        all_points = points
                    else:
                        all_points = np.vstack((all_points, points))
                        
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} point clouds, total: {all_points.shape[0]} points")
                    
                    if all_points.shape[0] >= max_points:
                        print(f"Reached maximum points limit of {max_points}. Stopping.")
                        break
                        
        except ImportError:
            print("Error: This script requires ROS 2 Python packages to be installed.")
            return
    
    if all_points is None or len(all_points) == 0:
        print(f"No point cloud data found on topic {config['topic']}")
        return
    
    print(f"Stacked {count} point clouds with a total of {all_points.shape[0]} points.")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    if config['visualize']:
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", mesh_show_back_face=False)
    
    o3d.io.write_point_cloud(config['output_path'], pcd)
    print(f"Saved stacked point cloud to {config['output_path']}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stack PointCloud2 messages from MCAP and save as PCD/PLY')
    parser.add_argument('--config', type=str, required=True, help='path to config')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    main(config)
