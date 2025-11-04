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
from mpl_toolkits.mplot3d import Axes3D

from slam_data_utils import LidarProjector, get_camera_pose, get_lidar2cam_transform, visualize_points, visualize_trajectories
from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.converter import str_to_cvt_class

# This file obtain the depth and camera poses for training/eval visual SLAM methods

def load_data_for_frame(dataset_path, idx, config, debug=False, device='cpu'):
    data = {}
    
    image_dir = config['image_dir']
    pc_dir = config['pointcloud_dir']
    image_base_dir = os.path.join(dataset_path, image_dir)

    if 'thermal' in image_dir:
        if 'processed' in image_dir:
            data['image'] = str_to_cvt_class['ThermalImage'].from_kitti(image_base_dir, idx, device=device)
        else:
            data['image'] = str_to_cvt_class['Thermal16bitImage'].from_kitti(image_base_dir, idx, device=device)
    else:
        data['image'] = str_to_cvt_class['Image'].from_kitti(image_base_dir, idx, device=device)
    
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
            data['pointcloud'] = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, idx, device=device)
            data['pointcloud'].pts = stacked_points
        else:
            data['pointcloud'] = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, idx, device=device)
        if debug:
            print(f"Stacking {start_idx} to {end_idx-1}")
            if len(points) > 0:
                print(f"Total {stacked_points.shape} points")
                visualize_points(stacked_points)
            else:
                print(f"No stacked points available, using current frame only")
    else:
        data['pointcloud'] = str_to_cvt_class['PointCloud'].from_kitti(pc_base_dir, idx, device=device)
    
    return data


def process_frame(idx, dataset_path, config, tf_manager, projector, debug=False):
    data = load_data_for_frame(dataset_path, idx, config, debug)
    
    if 'image' not in data or 'pointcloud' not in data:
        print(f"Missing data for index {idx}")
        return None, None, None, None
    
    timestamp = data['image'].stamp
    
    # purely for visualization
    if data['image'].image.max() <= 1:
        img = data['image'].image.cpu().numpy() * 255.
        img = img.astype(np.uint8)
    else:
        img = data['image'].image.cpu().numpy()
        img = img.astype(np.uint16)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)
    
    slam_points = data['pointcloud'].pts.cpu().numpy()
    if debug:
        print(f"Slam points: {slam_points.shape}")

    # option 1: directly use tf_manager
    point_frame = config['pointcloud_frame']
    cam_frame = config['camera_frame']
    if tf_manager.can_transform(cam_frame, point_frame, timestamp):
        T_sensor2cam = tf_manager.get_transform(cam_frame, point_frame, timestamp)
        T_sensor2cam = T_sensor2cam.transform.cpu().numpy()
    else:
        print(f"No transform found")
        return None, None, None, None
    
    # option 2: use specified extrinsics & vehicle-odom from tf_manager
    vehicle_frame = 'vehicle'
    if (tf_manager.can_transform(vehicle_frame, cam_frame, timestamp) and 
        tf_manager.can_transform(cam_frame, point_frame, timestamp)):
        tf_vehicle2cam = tf_manager.get_transform(cam_frame, vehicle_frame, timestamp)
        T_vehicle2cam = tf_vehicle2cam.transform.cpu().numpy()
        
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
        return None, None, None, None
    
    pose_config = config.copy()
    if pose_config['pose_in_ned']:
        pose_config['camera_frame'] = pose_config['camera_frame'] + '_ned'
    
    odom_7d, cam_7d, _ = get_camera_pose(dataset_path, tf_manager, pose_config, idx)
    
    return img, depth_map, odom_7d, cam_7d


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

def main(args):
    config = yaml.safe_load(open(args.config, 'r'))
    config['device'] = args.device
    
    tf_dir = os.path.join(args.dataset, "tf")
    if not os.path.exists(tf_dir):
        print(f"TF directory not found: {tf_dir}")
        return
    
    tf_manager = TfManager.from_kitti(args.dataset, device=args.device)
    if config['pose_in_ned']:
        print("Adding NED to tf manager")
        cam_optical_frame2ned = torch.tensor([0., 0., 0., -0.5, -0.5, -0.5, 0.5])
        tf_manager.add_static_tf(src_frame=config['camera_frame'], dst_frame=config['camera_frame']+'_ned', transform=cam_optical_frame2ned.numpy())
    print(f"TF TREE:\n{tf_manager.tf_tree}")
    
    sample_idx = args.idx if args.idx is not None else 0
    sample_img = cv2.imread(os.path.join(args.dataset, config['image_dir'], f"{sample_idx:08d}.png"))
    if sample_img is None:
        print(f"Sample image not found for index {sample_idx}")
        return

    if 'mask_path' in config and os.path.exists(config['mask_path']):
        vehicle_mask = cv2.imread(config['mask_path'])
        if len(vehicle_mask.shape) == 3:
            vehicle_mask = cv2.cvtColor(vehicle_mask, cv2.COLOR_BGR2GRAY)
        vehicle_mask = (vehicle_mask > 0).astype(np.uint8)
    else:
        vehicle_mask = None
    
    print(f"Dataset: {args.dataset}, image_dir: {config['image_dir']}, point_cloud_dir: {config['pointcloud_dir']}")
    lidar_files = sorted(glob.glob(os.path.join(args.dataset, config['pointcloud_dir'], "*.npy")))
    img_files = sorted(glob.glob(os.path.join(args.dataset, config['image_dir'], "*.png")))
    print(f"Found {len(lidar_files)} lidar files and {len(img_files)} image files")
    print(f"Using chain: {config['use_chain']}")

    projector = LidarProjector(img_width=sample_img.shape[1], img_height=sample_img.shape[0], max_depth=config['max_depth'])
    
    # single frame debug
    if args.idx is not None:
        img, depth_map, odom_7d, cam_7d = process_frame(args.idx, args.dataset, config, tf_manager, projector, args.debug)
        if vehicle_mask is not None and depth_map is not None:
            depth_map[vehicle_mask == 1] = 0
        
        if img is not None and depth_map is not None:
            projector.save_depth_image(depth_map, "depth.png", as_float16=True)
            
            if cam_7d is not None:
                poses = [cam_7d]
                save_poses_to_file(poses, ".")
            
            if args.debug:
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
    
    # full folder processing
    else:
        print(f"Processing folder")
        depth_save_dir = os.path.join(args.dataset, config['depth_output_dir'])
        pose_save_dir = os.path.join(args.dataset, config.get('pose_output_dir', 'camera_poses'))
        os.makedirs(depth_save_dir, exist_ok=True)
        os.makedirs(pose_save_dir, exist_ok=True)
        
        successful_frames = 0
        ts_list = [None] * len(img_files)
        pose_list = [None] * len(img_files)
        odom_poses = []
        camera_poses = []
        
        proc_frames = len(img_files) if args.seq_to is None else min(args.seq_to, len(img_files))
        
        for idx in tqdm(range(proc_frames)):
            depth_path = os.path.join(depth_save_dir, f"{idx:08d}.png")
            
            if args.resume and os.path.exists(depth_path):
                successful_frames += 1
                continue
            
            img, depth_map, odom_7d, cam_7d = process_frame(idx, args.dataset, config, tf_manager, projector)
            
            if vehicle_mask is not None and depth_map is not None:
                depth_map[vehicle_mask == 1] = 0
            
            if img is not None and depth_map is not None:
                projector.save_depth_image(depth_map, depth_path, as_float16=True)
                
                if cam_7d is not None:
                    pose_list[idx] = cam_7d
                    successful_frames += 1
                    
                    data = load_data_for_frame(args.dataset, idx, config, device=args.device)
                    if 'image' in data:
                        ts_list[idx] = data['image'].stamp
                    
                    if args.debug: #visualize full traj after processing
                        odom_poses.append(odom_7d)
                        camera_poses.append(cam_7d)
        
        save_poses_to_file(pose_list, pose_save_dir)
        save_timestamps_to_file(ts_list, pose_save_dir)
        
        if args.debug and len(odom_poses) > 0:
            print(f"Showing visualization with {len(odom_poses)} poses...")
            visualize_trajectories(odom_poses, camera_poses, config)
        
        print(f"Successfully processed {successful_frames}/{proc_frames} frames")
        print(f"Saved depth maps to {depth_save_dir}")
        print(f"Saved camera poses to {pose_save_dir}/data.txt")
        print(f"Saved timestamps to {pose_save_dir}/timestamps.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SLAM depth maps and camera poses from dataset")
    parser.add_argument("--dataset", type=str, 
                       default="/storage/datasets/yamaha/20250429/rough_rider_grass",
                       help="Path to dataset directory")
    parser.add_argument("--idx", type=int, default=None, help="Frame index to process. If none, process folder")
    parser.add_argument("--config", type=str, default="./config/kitti_config/get_slam_depth.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device for tensor operations")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed frame")
    parser.add_argument("--debug", action="store_true", help="Debug mode prints more and visualizes")
    parser.add_argument("--seq_to", type=int, default=None, help="Sequence to use for visualization")
    args = parser.parse_args()
    main(args)

