#!/usr/bin/env python3

import yaml
from scipy.spatial.transform import Rotation as R
from ros_torch_converter.tf_manager import TfManager

"""
get_transform(frame_a, frame_b, timestamp) returns the pose of frame_b expressed in
frame_a: the homogeneous transform T_{a<-b} that maps point from frame_b into frame_a.
p_a = T_{a<-b} @ p_b.

Same convention as ROS tf2 lookupTransform(target=frame_a, source=frame_b): the FIRST
arg is the reference / "into" frame, the SECOND is the frame whose pose you want.

Viking verified example:
    get_transform('novatel/imu_frame', 'multisense/left_camera_optical_frame_ned'),
    the camera's pose in the IMU frame, translation ~[0.42, 0.22, 0.01] m (the camera
    sits ~0.42 m ahead of the IMU).

Variable convention: T_target_source.
"""

def debug_tf_directions():
    dataset_path = "/drive/datasets/offroad/20250429/rough_rider_grass"
    
    tf_manager = TfManager.from_kitti(dataset_path, device='cpu')
    
    calib_path = "/home/tartandriver/tartandriver_ws/src/core/static_tf_publisher/config/offroad/yamaha.yaml"
    calib_config = yaml.safe_load(open(calib_path, 'r'))
    tf_manager.update_from_calib_config(calib_config)
    
    print("=== TF TREE STRUCTURE ===")
    print(tf_manager.tf_tree)
    print()
    
    timestamp = 1745955638.192

    print("=== EXPECTED Multisense Left EXTRINSICS ===")
    print(f"p: {[0.17265, -0.15227, 0.05708]}")
    print(f"q: {[0.55940, -0.54718, 0.44603, 0.43442]}")
    
    print("=== ACTUAL TF MANAGER RESULTS ===")
    tf_camera_vehicle = tf_manager.get_transform('multisense/left_camera_optical_frame', 'vehicle', timestamp)
    T_camera_vehicle = tf_camera_vehicle.transform.cpu().numpy()
    print(f"get_transform('multisense/left_camera_optical_frame', 'vehicle'):")
    print(f"p: {T_camera_vehicle[:3, 3]}")
    print(f"q: {R.from_matrix(T_camera_vehicle[:3, :3]).as_quat()}")
    

    print("=== EXPECTED Thermal Left EXTRINSICS ===")
    print(f"p: {[0.19265, -0.15227,  0.05708]}")
    print(f"q: {[0.5574714, -0.54525728,  0.44844936,  0.43682184]}")

    print("=== ACTUAL TF MANAGER RESULTS ===")
    tf_thermal_left_vehicle = tf_manager.get_transform('thermal_left/optical_frame', 'vehicle', timestamp)
    T_thermal_left_vehicle = tf_thermal_left_vehicle.transform.cpu().numpy()
    print(f"get_transform('thermal_left/optical_frame', 'vehicle'):")
    print(f"p: {T_thermal_left_vehicle[:3, 3]}")
    print(f"q: {R.from_matrix(T_thermal_left_vehicle[:3, :3]).as_quat()}")

    print(f"=== Thermal Right Extrinsics ===")
    print(f"p: {[0.19371392, -0.22469098, 0.3842142]}")
    print(f"q: {[0.5574714, -0.54525728,  0.44844936,  0.43682184]}")

    tf_thermal_right_vehicle = tf_manager.get_transform('thermal_right/optical_frame', 'vehicle', timestamp)
    T_thermal_right_vehicle = tf_thermal_right_vehicle.transform.cpu().numpy()
    print(f"get_transform('thermal_right/optical_frame', 'vehicle'):")
    print(f"p: {T_thermal_right_vehicle[:3, 3]}")
    print(f"q: {R.from_matrix(T_thermal_right_vehicle[:3, :3]).as_quat()}")

    # check convention of camera and odometry (e.g. NED or ENU). MACSLAM uses NED
    tf_imu_camera_ned = tf_manager.get_transform('novatel/imu_frame', 'multisense/left_camera_optical_frame_ned', timestamp)
    T_imu_camera_ned = tf_imu_camera_ned.transform.cpu().numpy()
    print(f"get_transform('novatel/imu_frame', 'multisense/left_camera_optical_frame_ned'):")
    print(f"p: {T_imu_camera_ned[:3, 3]}")
    print(f"q: {R.from_matrix(T_imu_camera_ned[:3, :3]).as_quat()}")

    tf_imu_thermal_ned = tf_manager.get_transform('novatel/imu_frame', 'thermal_left/optical_frame_ned', timestamp)
    T_imu_thermal_ned = tf_imu_thermal_ned.transform.cpu().numpy()
    print(f"get_transform('novatel/imu_frame', 'thermal_left/optical_frame_ned'):")
    print(f"p: {T_imu_thermal_ned[:3, 3]}")
    print(f"q: {R.from_matrix(T_imu_thermal_ned[:3, :3]).as_quat()}")

if __name__ == "__main__":
    debug_tf_directions()
