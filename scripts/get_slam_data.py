import numpy as np
import cv2
import torch
import glob
import os
from tqdm import tqdm
import argparse
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from slam_data_utils import LidarProjector, get_lidar2cam_transform, visualize_points, visualize_trajectories
from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.converter import str_to_cvt_class

# This file obtain the depth and camera poses for training/eval visual SLAM methods

def load_image_for_frame(dataset_path, idx, config, device='cpu'):
    image_dir = config['image_dir']
    image_base_dir = os.path.join(dataset_path, image_dir)

    if 'thermal' in image_dir:
        if 'processed' in image_dir:
            return str_to_cvt_class['ThermalImage'].from_kitti(image_base_dir, idx, device=device)
        else:
            return str_to_cvt_class['Thermal16bitImage'].from_kitti(image_base_dir, idx, device=device)
    else:
        return str_to_cvt_class['Image'].from_kitti(image_base_dir, idx, device=device)


def load_pointcloud_for_frame(dataset_path, idx, config, debug=False, device='cpu'):
    pc_dir = config['pointcloud_dir']
    pc_base_dir = os.path.join(dataset_path, pc_dir)
    if 'stack' in config:
        points = []
        start_idx = max(0, idx-config['stack'])
        end_idx = min(len(os.listdir(pc_base_dir)), idx+config['stack']+1)
        for i in range(start_idx, end_idx):
            pc_file = os.path.join(pc_base_dir, f"{i:08d}.npy")
            if os.path.exists(pc_file):
                pc = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, i, device=device)
                points.append(pc.pts)
            elif debug:
                print(f"Skipping missing pointcloud file: {pc_file}")
        if len(points) > 0:
            stacked_points = torch.cat(points, dim=0)
            pointcloud = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, idx, device=device)
            pointcloud.pts = stacked_points
        else:
            pointcloud = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, idx, device=device)
        if debug:
            print(f"Stacking {start_idx} to {end_idx-1}")
            if len(points) > 0:
                print(f"Total {stacked_points.shape} points")
                visualize_points(
                    stacked_points,
                    window_name=f"Pointcloud idx {idx} (stack {start_idx}-{end_idx-1}, {stacked_points.shape[0]} pts)",
                )
            else:
                print(f"No stacked points available, using current frame only")
    else:
        pointcloud = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, idx, device=device)

    return pointcloud


def process_depth(idx, dataset_path, config, tf_manager, projector, image_data, debug=False):
    pointcloud = load_pointcloud_for_frame(dataset_path, idx, config, debug)

    timestamp = image_data.stamp
    slam_points = pointcloud.pts.cpu().numpy()
    if debug:
        print(f"Slam points: {slam_points.shape}")

    point_frame = config['pointcloud_frame']
    cam_frame = config['camera_frame']
    if tf_manager.can_transform(cam_frame, point_frame, timestamp):
        T_sensor2cam = tf_manager.get_transform(cam_frame, point_frame, timestamp)
        T_sensor2cam = T_sensor2cam.transform.cpu().numpy()
    else:
        print(f"No transform found for depth at frame {idx}")
        return None

    vehicle_frame = 'vehicle'
    T_sensor2cam_chain = None
    if (tf_manager.can_transform(vehicle_frame, cam_frame, timestamp) and
        tf_manager.can_transform(cam_frame, point_frame, timestamp)):
        T_vehicle2cam = get_lidar2cam_transform(config['vehicle_2_cam'])
        tf_point2vehicle = tf_manager.get_transform(vehicle_frame, point_frame, timestamp)
        T_point2vehicle = tf_point2vehicle.transform.cpu().numpy()
        T_sensor2cam_chain = T_vehicle2cam @ T_point2vehicle

    intrinsics = config['intrinsics'][0]

    if config['use_chain']:
        depth_map = projector.project_lidar_to_image(slam_points, intrinsics, T_sensor2cam_chain)
    else:
        depth_map = projector.project_lidar_to_image(slam_points, intrinsics, T_sensor2cam)

    if debug:
        print(f"Depth max: {depth_map.max()}, min: {depth_map.min()}, mean: {depth_map.mean()}")

    if (depth_map == 0).all():
        print(f"No points projected to image for frame {idx}")
        return None

    return depth_map


def process_odom(timestamp, config, tf_manager):
    # Use tf_manager to handle sync and interpolation
    odom_frame = config['odometry_frame']
    cam_frame = config['camera_frame']
    if config.get('pose_in_ned', False):
        cam_frame = cam_frame + '_ned'

    if not tf_manager.can_transform(odom_frame, cam_frame, timestamp):
        print(f"Cannot get transform at timestamp {timestamp}")
        return None

    tf_cam = tf_manager.get_transform(odom_frame, cam_frame, timestamp)
    T_cam = tf_cam.transform.cpu().numpy()

    cam_pos = T_cam[:3, 3]
    cam_rot = R.from_matrix(T_cam[:3, :3]).as_quat()
    cam_7d = np.concatenate([cam_pos, cam_rot])

    return cam_7d


def image_to_viz(image_data):
    if image_data.image.max() <= 1:
        img = image_data.image.cpu().numpy() * 255.
        img = img.astype(np.uint8)
    else:
        img = image_data.image.cpu().numpy()
        img = img.astype(np.uint16)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)
    return img


def save_poses_to_file(poses, save_dir):
    """
    poses: list of 7-element numpy arrays [x, y, z, qx, qy, qz, qw]
    """
    os.makedirs(save_dir, exist_ok=True)
    pose_file = os.path.join(save_dir, "data.txt")

    pose_array = np.array([p if p is not None else np.full(7, np.nan) for p in poses])
    np.savetxt(pose_file, pose_array)

def save_timestamps_to_file(timestamps, save_dir):
    timestamp_file = os.path.join(save_dir, "timestamps.txt")
    with open(timestamp_file, 'w') as f:
        for i, timestamp in enumerate(timestamps):
            if timestamp is not None:
                f.write(f"{timestamp:.9f}\n")
            else:
                f.write("nan\n")

def process_single_config(config, args, tf_manager):
    extract_depth = args.depth
    extract_odom = args.odom
    config_name = config.get('pose_output_dir', config['image_dir'])

    sample_idx = args.idx[0] if args.idx is not None else 0
    sample_img = cv2.imread(os.path.join(args.dataset, config['image_dir'], f"{sample_idx:08d}.png"))
    if sample_img is None:
        print(f"[{config_name}] Sample image not found for index {sample_idx}")
        return

    if extract_depth and 'mask_path' in config and os.path.exists(config['mask_path']):
        vehicle_mask = cv2.imread(config['mask_path'])
        if len(vehicle_mask.shape) == 3:
            vehicle_mask = cv2.cvtColor(vehicle_mask, cv2.COLOR_BGR2GRAY)
        vehicle_mask = (vehicle_mask > 0).astype(np.uint8)
    else:
        vehicle_mask = None

    img_files = sorted(glob.glob(os.path.join(args.dataset, config['image_dir'], "*.png")))

    projector = None
    if extract_depth:
        pc_dir = os.path.join(args.dataset, config['pointcloud_dir'])
        if not os.path.exists(pc_dir):
            print(f"[{config_name}] Pointcloud directory not found: {pc_dir}. Skipping depth.")
            extract_depth = False
        else:
            lidar_files = sorted(glob.glob(os.path.join(pc_dir, "*.npy")))
            print(f"[{config_name}] Found {len(lidar_files)} lidar files")
            projector = LidarProjector(img_width=sample_img.shape[1], img_height=sample_img.shape[0], max_depth=config['max_depth'])

    print(f"[{config_name}] image_dir: {config['image_dir']}, {len(img_files)} images")
    print(f"[{config_name}] depth: {extract_depth}, odom: {extract_odom}")
    if extract_depth:
        print(f"[{config_name}] Using chain: {config['use_chain']}")

    # single frame / range debug
    if args.idx is not None:
        if len(args.idx) == 1:
            debug_indices = [args.idx[0]]
        else:
            start_idx, end_idx = args.idx[0], args.idx[-1]
            if end_idx >= len(img_files):
                print(f"[{config_name}] idx end {end_idx} out of bounds ({len(img_files)} frames), "
                      f"clamping to {len(img_files)-1}")
                end_idx = len(img_files) - 1
            debug_indices = list(range(start_idx, end_idx + 1))
        is_range = len(debug_indices) > 1

        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)

        camera_poses = []
        saved_outputs = []
        for idx in debug_indices:
            image_data = load_image_for_frame(args.dataset, idx, config, device=args.device)

            depth_map = None
            if extract_depth:
                depth_map = process_depth(idx, args.dataset, config, tf_manager, projector, image_data, args.verbose)
                if vehicle_mask is not None and depth_map is not None:
                    depth_map[vehicle_mask == 1] = 0
                if depth_map is not None:
                    depth_name = f"depth_{idx:08d}.png" if is_range else "depth.png"
                    depth_path = os.path.join(debug_dir, depth_name)
                    projector.save_depth_image(depth_map, depth_path, as_float16=True)
                    saved_outputs.append(depth_path)

            if extract_odom:
                cam_7d = process_odom(image_data.stamp, config, tf_manager)
                if cam_7d is not None:
                    camera_poses.append(cam_7d)

            if args.verbose and depth_map is not None:
                img = image_to_viz(image_data)
                depth_viz = projector.visualize_depth(depth_map)
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("Image")
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(depth_viz)
                plt.title("Viz Depth Map")
                plt.subplot(1, 3, 3)
                plt.imshow(depth_map, cmap='gray')
                plt.title("Raw Depth Map")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                projector.visualize_rgbd(img, depth_map, "Depth Projection", point_size=1)

        if extract_odom and len(camera_poses) > 0:
            save_poses_to_file(camera_poses, debug_dir)
            saved_outputs.append(os.path.join(debug_dir, "data.txt"))
            if is_range and args.verbose:
                print(f"[{config_name}] Visualizing {len(camera_poses)} camera poses "
                      f"over idx {debug_indices[0]}..{debug_indices[-1]}")
                visualize_trajectories([], camera_poses, config)

        if saved_outputs:
            print(f"[{config_name}] Saved {len(saved_outputs)} debug output(s) to {os.path.abspath(debug_dir)}:")
            for name in saved_outputs:
                print(f"  - {os.path.basename(name)}")

    # full folder processing
    else:
        depth_save_dir = None
        pose_save_dir = None
        if extract_depth:
            depth_save_dir = os.path.join(args.dataset, config['depth_output_dir'])
            os.makedirs(depth_save_dir, exist_ok=True)
        if extract_odom:
            pose_save_dir = os.path.join(args.dataset, config.get('pose_output_dir', 'camera_poses'))
            os.makedirs(pose_save_dir, exist_ok=True)

        successful_depth = 0
        successful_odom = 0
        ts_list = [None] * len(img_files)
        pose_list = [None] * len(img_files)
        odom_poses = []
        camera_poses = []

        if args.verbose and extract_odom:
            odom_data_file = os.path.join(args.dataset, "odometry", "data.txt")
            if os.path.exists(odom_data_file):
                odom_all = np.loadtxt(odom_data_file).reshape(-1, 13)[:, :7]
                odom_poses = list(odom_all)
                print(f"[{config_name}] Loaded {len(odom_poses)} odometry poses for debug viz")

        proc_frames = len(img_files) if args.seq_to is None else min(args.seq_to, len(img_files))

        for idx in tqdm(range(proc_frames), desc=config_name):
            image_data = load_image_for_frame(args.dataset, idx, config, device=args.device)

            if extract_depth:
                depth_path = os.path.join(depth_save_dir, f"{idx:08d}.png")
                if args.resume and os.path.exists(depth_path):
                    successful_depth += 1
                else:
                    depth_map = process_depth(idx, args.dataset, config, tf_manager, projector, image_data, args.verbose)
                    if vehicle_mask is not None and depth_map is not None:
                        depth_map[vehicle_mask == 1] = 0
                    if depth_map is not None:
                        projector.save_depth_image(depth_map, depth_path, as_float16=True)
                        successful_depth += 1

            if extract_odom:
                cam_7d = process_odom(image_data.stamp, config, tf_manager)
                if cam_7d is not None:
                    pose_list[idx] = cam_7d
                    ts_list[idx] = image_data.stamp
                    successful_odom += 1

                    if args.verbose:
                        camera_poses.append(cam_7d)

        if extract_odom:
            save_poses_to_file(pose_list, pose_save_dir)
            save_timestamps_to_file(ts_list, pose_save_dir)

        if args.verbose and len(camera_poses) > 0:
            print(f"Showing visualization with {len(camera_poses)} camera poses, {len(odom_poses)} odom poses...")
            visualize_trajectories(odom_poses, camera_poses, config)

        if extract_depth:
            print(f"[{config_name}] Depth: {successful_depth}/{proc_frames} frames → {depth_save_dir}")
        if extract_odom:
            print(f"[{config_name}] Odom: {successful_odom}/{proc_frames} frames → {pose_save_dir}/data.txt")


def main(args):
    if not args.depth and not args.odom:
        print("Nothing to do. Specify --depth and/or --odom.")
        return

    tf_dir = os.path.join(args.dataset, "tf")
    if not os.path.exists(tf_dir):
        print(f"TF directory not found: {tf_dir}")
        return

    configs = []
    for config_path in args.config:
        config = yaml.safe_load(open(config_path, 'r'))
        configs.append(config)

    tf_manager = TfManager.from_kitti(args.dataset, device=args.device)

    if args.odom:
        cam_optical_frame2ned = torch.tensor([0., 0., 0., -0.5, -0.5, -0.5, 0.5])
        added_ned_frames = set()
        for config in configs:
            if config.get('pose_in_ned', False):
                cam_frame = config['camera_frame']
                if cam_frame not in added_ned_frames:
                    print(f"Adding NED transform for {cam_frame}")
                    tf_manager.add_static_tf(src_frame=cam_frame, dst_frame=cam_frame+'_ned', transform=cam_optical_frame2ned.numpy())
                    added_ned_frames.add(cam_frame)

    print(f"Dataset: {args.dataset}")
    print(f"Processing {len(configs)} config(s): {[os.path.basename(p) for p in args.config]}")
    print(f"---")

    for config_path, config in zip(args.config, configs):
        print(f"\n=== {os.path.basename(config_path)} ===")
        process_single_config(config, args, tf_manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SLAM depth maps and camera poses from dataset")
    parser.add_argument("--dataset", type=str,
                       default="/storage/datasets/yamaha/20250429/rough_rider_grass",
                       help="Path to dataset directory")
    parser.add_argument("--config", type=str, nargs='+',
                       default=["./config/kitti_config/get_slam_depth.yaml"],
                       help="Path to config file(s). Pass multiple for multi-camera processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device for tensor operations")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed frame")
    parser.add_argument("--seq_to", type=int, default=None, help="Process only first N frames of the folder (default: all)")
    parser.add_argument("--depth", action="store_true", help="Extract depth maps (requires pointcloud data)")
    parser.add_argument("--odom", action="store_true", help="Extract camera poses (only needs odometry + TF)")
    # debug mode
    parser.add_argument("--idx", type=int, nargs='+', default=None,
                       help="Frame index/indices to debug. One value debugs a single frame; "
                            "two values [start end] debug an inclusive range and visualize the odom trajectory.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode prints more and visualizes")

    args = parser.parse_args()
    main(args)
