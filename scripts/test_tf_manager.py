import time
import yaml
from ros_torch_converter.tf_manager import TfManager
from tartandriver_utils.os_utils import load_yaml
import numpy as np
import os
from tartandriver_utils.geometry_utils import pose_to_htm, htm_to_pose, TrajectoryInterpolator

# general helper functions
def tf_manager_from_rosbag(bag_fp, odometry_configs=[], tf_inversions=[]):
    tf_manager = TfManager.from_rosbag(bag_fp, odometry_configs, tf_inversions, device='cuda')
    return tf_manager

def tf_manager_from_kitti(kitti_fp):
    return TfManager.from_kitti(kitti_fp)

def print_trees(tf_manager):
    print('VEHICLE TREE:')
    print(tf_manager.vehicle_tree)
    print('ODOM TREES:')
    for name, tree in tf_manager.odom_trees.items():
        print("----------------------")
        print(f"Tree name: {name}")
        print(tree)
        print("----------------------")
    print('BRIDGES:')
    for name, bridge in tf_manager.bridges.items():
        print("----------------------")
        print(f"Bridge name: {name}")
        print(bridge)
        print("----------------------")

# from_rosbag tests
def test_vehicle_tree_no_config(tf_inversions):
    # bag_fp = "/home/tartandriver/rosbags/20260617/00_lidar_thermal_calib_1"
    bag_fp = "/home/tartandriver/rosbags/20251120/teleop/01_rtk_collect_warehouse_2"
    tf_manager = tf_manager_from_rosbag(bag_fp, tf_inversions=tf_inversions)
    print_trees(tf_manager)

def test_vehicle_tree_config(config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20260617/00_lidar_thermal_calib_1"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    print_trees(tf_manager)

def test_multiple_odom_config(config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20260716/teleop/00_sensor_coverage_01"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    print_trees(tf_manager)

def test_odom_no_config(tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20260709/04_warehouse_calib_5"
    tf_manager = tf_manager_from_rosbag(bag_fp, tf_inversions=tf_inversions)
    print_trees(tf_manager)

def test_no_odom_config(config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20250125/20250125_day"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    print_trees(tf_manager)

def test_one_odom_config(config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20260709/04_warehouse_calib_5"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    print_trees(tf_manager)

# update_from_calib_config_tests
def test_update_from_config_vehicle_tree_default(odom_config, calib_config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20251028/auto/2_wfig8_forwards_no_slope"
    tf_manager = tf_manager_from_rosbag(bag_fp, odom_config, tf_inversions)
    tf_manager.update_from_calib_config(calib_config)
    print_trees(tf_manager)

def test_update_from_config_vehicle_tree_explicit(odom_config, calib_config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20251028/auto/2_wfig8_forwards_no_slope"
    tf_manager = tf_manager_from_rosbag(bag_fp, odom_config, tf_inversions)
    tf_manager.update_from_calib_config(calib_config, tree=tf_manager.vehicle_tree)
    print_trees(tf_manager)

def test_update_from_config_odom_tree(odom_config, calib_config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20260716/teleop/00_sensor_coverage_01"
    tf_manager = tf_manager_from_rosbag(bag_fp, odom_config, tf_inversions)
    tf_manager.update_from_calib_config(calib_config, tree=tf_manager.odom_trees['super_odometry'])
    print_trees(tf_manager)

def test_update_from_config_vehicle_and_odom_trees(odom_config, vehicle_calib_config, odom_calib_config, tf_inversions):
    bag_fp = "/home/tartandriver/rosbags/20260716/teleop/00_sensor_coverage_01"
    tf_manager = tf_manager_from_rosbag(bag_fp, odom_config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(odom_calib_config, tree=tf_manager.odom_trees['super_odometry'])
    print_trees(tf_manager)

# to_kitti tests
def test_correct_file_format_vehicle_only(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260617/00_lidar_thermal_calib_1"
    kitti_out_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_lidar_thermal_calib_1_kitti"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])
    tf_manager.to_kitti(kitti_out_fp)

def test_correct_file_format_one_odom(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260709/04_warehouse_calib_5"
    kitti_out_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/04_warehouse_calib_5_kitti"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])
    tf_manager.to_kitti(kitti_out_fp)

def test_correct_file_format_mult_odom(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260716/teleop/00_sensor_coverage_01"
    kitti_out_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])
    tf_manager.to_kitti(kitti_out_fp)

# from_kitti tests
def test_successful_load_from_kitti_vehicle_only():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_lidar_thermal_calib_1_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    print_trees(tf_manager)

def test_successful_load_from_kitti_one_odom():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/04_warehouse_calib_5_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    print_trees(tf_manager)

def test_successful_load_from_kitti_mult_odom():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    print_trees(tf_manager)

def test_lookup_from_kitti_vehicle_only():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_lidar_thermal_calib_1_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    tmin, tmax = tf_manager.get_valid_times('vehicle', 'multisense/left_camera_optical_frame')
    transform = tf_manager.get_transform('multisense/left_camera_optical_frame', 'vehicle', tmin)
    pose_exp = htm_to_pose(transform.transform.cpu().numpy())
    pose_act = np.array([0.17265, -0.15227, 0.05708, 0.55940, -0.54718, 0.44603, 0.43442])
    print(f"Expected: {pose_exp} actual {pose_act}")

def test_lookup_from_kitti_one_odom():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/04_warehouse_calib_5_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)

    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    transforms = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'transforms.txt'))
    tinterp = TrajectoryInterpolator(times, transforms)

    idx = len(times) // 2
    ts = times[idx]
    earth_vehicle_pose = tinterp(ts)

    transform = tf_manager.get_transform('thermal_left/optical_frame', 'earth', ts)
    pose_exp = htm_to_pose(transform.transform.cpu().numpy())

    earth_vehicle_htm = pose_to_htm(earth_vehicle_pose)
    vehicle_thermal_left_pose = np.array([-0.08925464, 0.1890916, -0.14084132, 0.5574714, -0.54525728, 0.44844936, -0.43682184])
    vehicle_thermal_left_htm = pose_to_htm(vehicle_thermal_left_pose)
    htm_act = np.linalg.inv(earth_vehicle_htm @ vehicle_thermal_left_htm)
    pose_act = htm_to_pose(htm_act)

    print(f"Expected: {pose_exp}")
    print(f"Actual: {pose_act}")

def test_lookup_from_kitti_mult_odom():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)

    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))
    transforms = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'transforms.txt'))
    tinterp = TrajectoryInterpolator(times, transforms)

    idx = len(times) // 2
    ts = times[idx]
    sensor_init_vehicle_pose = tinterp(ts)
    
    transform = tf_manager.get_transform('multisense/head', 'sensor_init', ts)
    pose_exp = htm_to_pose(transform.transform.cpu().numpy())
    
    sensor_init_vehicle_htm = pose_to_htm(sensor_init_vehicle_pose)
    vehicle_multisense_head_pose = np.array([-0.0949,  0.0641, -0.136, -0.0119,  0.1131,  0.0003,  0.9935])
    vehicle_multisense_head_htm = pose_to_htm(vehicle_multisense_head_pose)
    htm_act = np.linalg.inv(sensor_init_vehicle_htm @ vehicle_multisense_head_htm)
    pose_act = htm_to_pose(htm_act)

    print(f"Expected: {pose_exp}")
    print(f"Actual: {pose_act}")

def test_lookup_from_kitti_undefined():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))

    idx = len(times) // 2
    ts = times[idx]

    try:
        trange = tf_manager.get_transform("earth", "sensor_init_rot", ts)
    except AssertionError as e:
        print("Successfully threw assertion error!")
        print(f"Error: {e}")
    else:
        AssertionError("Expected assertion to fail since earth and sensor_init are part of two different odometry sources")

# get_transform tests
def test_vehicle_tree_lookup(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260617/00_lidar_thermal_calib_1"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])
    tmin, tmax = tf_manager.get_valid_times('vehicle', 'multisense/left_camera_optical_frame')
    transform = tf_manager.get_transform('multisense/left_camera_optical_frame', 'vehicle', tmin)
    pose_exp = htm_to_pose(transform.transform.cpu().numpy())
    pose_act = np.array([0.17265, -0.15227, 0.05708, 0.55940, -0.54718, 0.44603, 0.43442])
    print(f"Expected: {pose_exp} actual {pose_act}")

def test_defined_cross_tree_lookup_one_odom(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260709/04_warehouse_calib_5"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])

    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/04_warehouse_calib_5_kitti"
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    transforms = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'transforms.txt'))
    tinterp = TrajectoryInterpolator(times, transforms)

    idx = len(times) // 2
    ts = times[idx]
    earth_vehicle_pose = tinterp(ts)

    transform = tf_manager.get_transform('thermal_left/optical_frame', 'earth', ts)
    pose_exp = htm_to_pose(transform.transform.cpu().numpy())

    earth_vehicle_htm = pose_to_htm(earth_vehicle_pose)
    vehicle_thermal_left_pose = np.array([-0.08925464, 0.1890916, -0.14084132, 0.5574714, -0.54525728, 0.44844936, -0.43682184])
    vehicle_thermal_left_htm = pose_to_htm(vehicle_thermal_left_pose)
    htm_act = np.linalg.inv(earth_vehicle_htm @ vehicle_thermal_left_htm)
    pose_act = htm_to_pose(htm_act)

    print(f"Expected: {pose_exp}")
    print(f"Actual: {pose_act}")

def test_defined_cross_tree_lookup_mult_odom(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260716/teleop/00_sensor_coverage_01"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])

    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))
    transforms = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'transforms.txt'))
    tinterp = TrajectoryInterpolator(times, transforms)

    idx = len(times) // 2
    ts = times[idx]
    sensor_init_vehicle_pose = tinterp(ts)
    
    transform = tf_manager.get_transform('multisense/head', 'sensor_init', ts)
    pose_exp = htm_to_pose(transform.transform.cpu().numpy())
    
    sensor_init_vehicle_htm = pose_to_htm(sensor_init_vehicle_pose)
    vehicle_multisense_head_pose = np.array([-0.0949,  0.0641, -0.136, -0.0119,  0.1131,  0.0003,  0.9935])
    vehicle_multisense_head_htm = pose_to_htm(vehicle_multisense_head_pose)
    htm_act = np.linalg.inv(sensor_init_vehicle_htm @ vehicle_multisense_head_htm)
    pose_act = htm_to_pose(htm_act)

    print(f"Expected: {pose_exp}")
    print(f"Actual: {pose_act}")

def test_undefined_cross_tree_lookup(config, tf_inversions, vehicle_calib_config, so_calib_config):
    bag_fp = "/home/tartandriver/rosbags/20260716/teleop/00_sensor_coverage_01"
    tf_manager = tf_manager_from_rosbag(bag_fp, config, tf_inversions)
    tf_manager.update_from_calib_config(vehicle_calib_config, tree=tf_manager.vehicle_tree)
    tf_manager.update_from_calib_config(so_calib_config, tree=tf_manager.odom_trees['super_odometry'])

    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))

    idx = len(times) // 2
    ts = times[idx]

    try:
        trange = tf_manager.get_transform("earth", "sensor_init_rot", ts)
    except AssertionError as e:
        print("Successfully threw assertion error!")
        print(f"Error: {e}")
    else:
        AssertionError("Expected assertion to fail since earth and sensor_init are part of two different odometry sources")

# get_valid_times tests
def test_vehicle_get_valid_times():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_lidar_thermal_calib_1_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    # test that get_valid_times with frame1 == frame2 returns -inf, inf
    print(f"Testing get_valid_times with frame1 == frame2 == vehicle")
    tmin, tmax = tf_manager.get_valid_times("vehicle", "vehicle")
    print(f"tmin: {tmin}, tmax: {tmax}")
    
    # test that get_valid_times with both frames in the vehicle tree return -inf, inf 
    # (since all nodes in the vehicle tree represent static tfs)
    print(f"Testing get_valid_times with two frames in vehicle tree, frame1 != frame2")
    tmin, tmax = tf_manager.get_valid_times("vehicle", "gq7_imu_link")
    print(f"tmin: {tmin}, tmax: {tmax}")

def test_defined_cross_tree_get_valid_times_one_odom():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/04_warehouse_calib_5_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    # test that get_valid_times with one frame in the vehicle tree and one frame in an odom tree returns
    # correct time range (the intersection of the time ranges of the transforms on the path between the two frames)
    print(f"Testing get_valid_times with one frame in vehicle tree and one frame in rtk tree")
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    node = tf_manager.odom_trees['rtk'].nodes['vehicle_rtk']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times("vehicle", "earth")
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, "Expected and actual tranges don't match"
    print("Expected trange matches actual!")

def test_defined_cross_tree_get_valid_times_mult_odom():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    # same test as test_defined_cross_tree_get_valid_times_one_odom (but using super odometry tf), but this tf_manager object has both 
    # super odometry and rtk odom
    print(f"------Testing get_valid_times with one frame in vehicle tree and one frame in super_odometry tree-----")
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))
    node = tf_manager.odom_trees['super_odometry'].nodes['vehicle_so']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times("vehicle", "sensor_init")
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, "Expected and actual tranges don't match"
    print("Expected trange matches actual!")
    print()
    print(f"-----Testing get_valid_times with one frame in vehicle tree and one frame in rtk tree-----")
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    node = tf_manager.odom_trees['rtk'].nodes['vehicle_rtk']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times("vehicle", "earth")
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, "Expected and actual tranges don't match"
    print("Expected trange matches actual!")

def test_undefined_cross_tree_get_valid_times():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    try:
        trange = tf_manager.get_valid_times("earth", "sensor_init")
    except AssertionError as e:
        print("Successfully threw assertion error!")
        print(f"Error: {e}")
    else:
        AssertionError("Expected assertion to fail since earth and sensor_init are part of two different odometry sources")

# get_valid_times_from_list tests
def test_frame_list_vehicle_only():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_lidar_thermal_calib_1_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    # returned range should be -inf, inf
    frame_list = ['gq7_imu_link', 'vehicle', 'multisense/left_camera_optical_frame', 'thermal_left/optical_frame', 'velodyne_2']
    act_tmin, act_tmax = tf_manager.get_valid_times_from_list(frame_list)
    exp_tmin = -float('inf')
    exp_tmax = float('inf')
    assert act_tmin == exp_tmin and act_tmax == exp_tmax, f"Expected trange {exp_tmin}, {exp_tmax} but got {act_tmin}, {act_tmax}"
    print("Expected trange matches actual!")

def test_frame_list_one_odom_and_vehicle():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/04_warehouse_calib_5_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    frame_list = ['earth', 'gq7_imu_link', 'gq7_map_ned', 'velodyne_1']
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    node = tf_manager.odom_trees['rtk'].nodes['vehicle_rtk']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times_from_list(frame_list)
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, f"Expected trange {exp_tmin}, {exp_tmax} but got {act_tmin}, {act_tmax}"
    print("Expected trange matches actual!")

def test_frame_list_multiple_odom_and_vehicle():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    # same test as test_frame_list_one_odom_and_vehicle, but this tf_manager object has both
    # super odometry and rtk odom
    print(f"------Testing get_valid_times_from_list with at least one frame in vehicle tree and one frame in super_odometry tree-----")
    frame_list_so = ['sensor_init', 'vehicle_so', 'vehicle', 'velodyne_1']
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))
    node = tf_manager.odom_trees['super_odometry'].nodes['vehicle_so']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times_from_list(frame_list_so)
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, f"Expected trange {exp_tmin}, {exp_tmax} but got {act_tmin}, {act_tmax}"
    print("Expected trange matches actual!")
    print()
    print(f"-----Testing get_valid_times_from_list with one frame in vehicle tree and one frame in rtk tree-----")
    frame_list_rtk = ['earth', 'vehicle_rtk', 'vehicle', 'velodyne_1']
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    node = tf_manager.odom_trees['rtk'].nodes['vehicle_rtk']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times_from_list(frame_list_rtk)
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, f"Expected trange {exp_tmin}, {exp_tmax} but got {act_tmin}, {act_tmax}"
    print("Expected trange matches actual!")

def test_frame_list_multiple_odom_only():
    kitti_fp = "/home/tartandriver/workspace/odom_tf_tree_changes/kitti_tests/00_sensor_coverage_01_kitti"
    tf_manager = tf_manager_from_kitti(kitti_fp)
    # same test as test_frame_list_multiple_odom_and_vehicle, but frame lists exclude frames in vehicle tree
    print(f"------Testing get_valid_times_from_list with at least one frame in vehicle tree and one frame in super_odometry tree-----")
    frame_list_so = ['sensor_init', 'vehicle_so']
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'sensor_init_to_vehicle_so', 'timestamps.txt'))
    node = tf_manager.odom_trees['super_odometry'].nodes['vehicle_so']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times_from_list(frame_list_so)
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, f"Expected trange {exp_tmin}, {exp_tmax} but got {act_tmin}, {act_tmax}"
    print("Expected trange matches actual!")
    print()
    print(f"-----Testing get_valid_times_from_list with one frame in vehicle tree and one frame in rtk tree-----")
    frame_list_rtk = ['earth', 'vehicle_rtk']
    times = np.loadtxt(os.path.join(kitti_fp, 'tf', 'earth_to_vehicle_rtk', 'timestamps.txt'))
    node = tf_manager.odom_trees['rtk'].nodes['vehicle_rtk']
    exp_tmin = times.min() - node.interp._tol
    exp_tmax = times.max() + node.interp._tol
    act_tmin, act_tmax = tf_manager.get_valid_times_from_list(frame_list_rtk)
    print(f"Expected range: {exp_tmin}, {exp_tmax}")
    print(f"Actual range: tmin: {act_tmin}, tmax: {act_tmax}")
    assert exp_tmin == act_tmin and exp_tmax == act_tmax, f"Expected trange {exp_tmin}, {exp_tmax} but got {act_tmin}, {act_tmax}"
    print("Expected trange matches actual!")

if __name__ == '__main__':
    bag_fp = "/media/striest/offroad/rosbags/20250508/teleop/power_tower_sidehill6/"
    calib_fp = "/home/tartandriver/tartandriver_ws/src/core/static_tf_publisher/config/offroad/yamaha.yaml"
    so_calib_fp = '/home/tartandriver/tartandriver_ws/src/core/static_tf_publisher/config/offroad/super_odometry.yaml'
    kitti_cfg_cp = "/home/tartandriver/tartandriver_ws/src/core/ros_torch_converter/config/kitti_config/odometry_tf.yaml"

    calib_config = load_yaml(calib_fp)
    so_calib_config = load_yaml(so_calib_fp)
    kitti_config = load_yaml(kitti_cfg_cp)

    odometry_configs = kitti_config['odometry_tf']
    tf_inversions = kitti_config['tf_inversions']

    # from_rosbag tests
    # test_vehicle_tree_no_config(tf_inversions)
    # test_vehicle_tree_config(odometry_configs, tf_inversions)
    # test_multiple_odom_config(odometry_configs, tf_inversions)
    # test_odom_no_config(tf_inversions)
    # test_no_odom_config(odometry_configs, tf_inversions)
    # test_one_odom_config(odometry_configs, tf_inversions)

    # update_from_calib_config tests
    # test_update_from_config_vehicle_tree_default(odometry_configs, calib_config, tf_inversions)
    # test_update_from_config_vehicle_tree_explicit(odometry_configs, calib_config, tf_inversions)
    # test_update_from_config_odom_tree(odometry_configs, so_calib_config, tf_inversions)
    # test_update_from_config_vehicle_and_odom_trees(odometry_configs, calib_config, so_calib_config, tf_inversions)

    # get_transform tests
    # test_vehicle_tree_lookup(odometry_configs, tf_inversions, calib_config, so_calib_config)
    # test_defined_cross_tree_lookup_one_odom(odometry_configs, tf_inversions, calib_config, so_calib_config)
    # test_defined_cross_tree_lookup_mult_odom(odometry_configs, tf_inversions, calib_config, so_calib_config)
    # test_undefined_cross_tree_lookup(odometry_configs, tf_inversions, calib_config, so_calib_config)

    # get_valid_times tests
    # test_vehicle_get_valid_times()
    # test_defined_cross_tree_get_valid_times_one_odom()
    # test_defined_cross_tree_get_valid_times_mult_odom()
    # test_undefined_cross_tree_get_valid_times()

    # get_valid_times_from_list tests
    # test_frame_list_vehicle_only()
    # test_frame_list_one_odom_and_vehicle()
    # test_frame_list_multiple_odom_and_vehicle()
    # test_frame_list_multiple_odom_only()

    # to_kitti tests
    # test_correct_file_format_vehicle_only(odometry_configs, tf_inversions, calib_config, so_calib_config)
    # test_correct_file_format_one_odom(odometry_configs, tf_inversions, calib_config, so_calib_config)
    # test_correct_file_format_mult_odom(odometry_configs, tf_inversions, calib_config, so_calib_config)

    # from_kitti tests
    # test_successful_load_from_kitti_vehicle_only()
    # test_successful_load_from_kitti_one_odom()
    # test_successful_load_from_kitti_mult_odom()
    # test_lookup_from_kitti_vehicle_only()
    # test_lookup_from_kitti_one_odom()
    # test_lookup_from_kitti_mult_odom()
    # test_lookup_from_kitti_undefined()