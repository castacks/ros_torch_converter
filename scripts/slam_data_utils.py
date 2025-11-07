import numpy as np
import os
import cv2
import yaml
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from ros_torch_converter.converter import str_to_cvt_class

def get_camera_pose(dataset, tf_manager, config, idx):
    try:
        odom_dir = os.path.join(dataset, "odometry")
        if not os.path.exists(odom_dir):
            raise FileNotFoundError(f"Odometry directory not found: {odom_dir}")
        
        odom_data = str_to_cvt_class['OdomRBState'].from_kitti(odom_dir, idx, device=config['device'])
        odom_7d = odom_data.state.cpu().numpy()[:7]
        timestamp = odom_data.stamp

        odom_frame = config['odometry_frame']
        cam_frame = config['camera_frame']
        
        if not tf_manager.can_transform(odom_frame, cam_frame, timestamp):
            print(f"Cannot get transforms at timestamp {timestamp}")
            return None, None, None
        
        tf_cam = tf_manager.get_transform(odom_frame, cam_frame, timestamp)
        T_cam = tf_cam.transform.cpu().numpy()
        
        cam_pos = T_cam[:3, 3]
        cam_rot = R.from_matrix(T_cam[:3, :3]).as_quat()
        cam_7d = np.concatenate([cam_pos, cam_rot])
        
        return odom_7d, cam_7d, timestamp
        
    except Exception as e:
        print(f"Error getting poses at idx {idx}: {e}")
        return None, None, None


class LidarProjector:
    """
    General-purpose class for projecting LiDAR points onto images.
    
    This class loads lidar points, projects them onto an image plane using 
    camera intrinsics and extrinsics, and generates a depth image.
    """
    
    def __init__(self, img_width=1920, img_height=1080, max_depth=None):
        """
        Initialize the LidarProjector.
        
        Args:
            img_width: Width of the output image
            img_height: Height of the output image
            max_depth: Maximum depth value to consider (in meters)
        """
        self.img_width = img_width
        self.img_height = img_height
        self.max_depth = max_depth
    
    def load_lidar(self, lidar_path, dtype='<f4', dim=4):
        """
        Load LiDAR points
        
        Args:
            lidar_path: Path to LiDAR binary file
            dtype: Data type of LiDAR points
            dim: Dimension of LiDAR points (usually 4 for x,y,z,intensity)
            
        Returns:
            lidar_points: Nx3 numpy array of LiDAR points
        """
        if lidar_path.endswith('.bin'):
            lidar_points = np.fromfile(lidar_path, dtype='<f4').reshape(-1, dim)
        elif lidar_path.endswith('.npy'):
            lidar_points = np.load(lidar_path)
        elif lidar_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(lidar_path)
            lidar_points = np.asarray(pcd.points)
        else:
            raise ValueError(f"Unsupported LiDAR file format: {lidar_path}. Rewrite lodar_lidar.")
        return lidar_points
    
    def load_calibration(self, calib, cam_name='cam0'):
        """
        Load calibration data from a file or array.
        
        Args:
            calib: Path to calibration file or calibration array
            cam_name: Name of the camera in calibration file
            
        Returns:
            intrinsics: Camera intrinsics as np.array([fx, fy, cx, cy])
        """
        if isinstance(calib, np.ndarray):
            return calib
        elif isinstance(calib, list):
            return np.array(calib)
        
        with open(calib, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        if cam_name in calib_data: # Kalibr format
            intrinsics = calib_data[cam_name]['intrinsics']
            return np.array([intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]])
        elif 'K' in calib_data: # OpenCV format
            K = np.array(calib_data['K']).reshape(3, 3)
            return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        elif 'intrinsics' in calib_data: # offroad format
            K = calib_data['intrinsics']['K']
            return np.array([K[0], K[4], K[2], K[5]])
        else:
            raise ValueError(f"Unrecognized calibration format in {calib}. Rewrite load_calibration() to handle this format.")
    
    
    def project_lidar_to_image(self, lidar_points, intrinsics, T_lidar2cam=None):
        """
        Project LiDAR points onto an image.
        
        Args:
            lidar_points: Nx4 or Nx3 numpy array of LiDAR points
            intrinsics: Camera intrinsics as [fx, fy, cx, cy]
            T_lidar2cam: 4x4 transformation matrix from LiDAR to camera
            
        Returns:
            depth_map: HxW depth image (np.float32)
        """
        if lidar_points.shape[1] < 3:
            raise ValueError("LiDAR points must have at least xyz coordinates")
        if T_lidar2cam is None:
            raise ValueError("Transformation matrix from LiDAR to camera is required")
        points_xyz = lidar_points[:, :3]
        points_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
        
        points_cam = (T_lidar2cam @ points_homogeneous.T).T
        
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]
        
        # Keep only points in front of camera (and within max depth)
        if self.max_depth is not None:
            valid_points = (Z > 0) & (Z < self.max_depth)
        else:
            valid_points = Z > 0
        X = X[valid_points]
        Y = Y[valid_points]
        Z = Z[valid_points]
        
        if len(Z) == 0:
            print("Warning: No valid points after filtering")
            return np.zeros((self.img_height, self.img_width), dtype=np.float32)
        

        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])

        # Project to 2D image
        uv_homogeneous = K @ np.vstack((X, Y, Z))
        u = uv_homogeneous[0] / uv_homogeneous[2]
        v = uv_homogeneous[1] / uv_homogeneous[2]
        
        valid_bounds = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)
        u = u[valid_bounds]
        v = v[valid_bounds] 
        Z = Z[valid_bounds]
        
        # Convert to integer pixel coordinates
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        depth_map = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        
        for i in range(len(u_int)):
            if 0 <= u_int[i] < self.img_width and 0 <= v_int[i] < self.img_height: # Check within image bounds
                if depth_map[v_int[i], u_int[i]] == 0:
                    depth_map[v_int[i], u_int[i]] = Z[i]
                else: # Keep the closest point
                    depth_map[v_int[i], u_int[i]] = min(depth_map[v_int[i], u_int[i]], Z[i])
        
        return depth_map
    
    def project_and_merge_multiple_scans(self, lidar_paths, intrinsics, T_lidar2cam=None, filter_points=True):
        """
        Project multiple LiDAR scans and merge them into a single depth map.
        
        Args:
            lidar_paths: List of paths to LiDAR files
            intrinsics: Camera intrinsics as [fx, fy, cx, cy]
            T_lidar2cam: 4x4 transformation matrix from LiDAR to camera
            filter_points: Whether to filter points based on max_depth and positivity
            
        Returns:
            depth_map: HxW depth image
        """
        combined_depth = None
        
        for lidar_path in lidar_paths:
            lidar_points = self.load_lidar(lidar_path)
            
            depth_map = self.project_lidar_to_image(lidar_points, intrinsics, T_lidar2cam, filter_points)
            
            # Merge with previous depth maps
            if combined_depth is None:
                combined_depth = depth_map
            else:
                # Keep the closest point at each pixel
                valid_pixels = depth_map > 0
                update_pixels = valid_pixels & ((combined_depth == 0) | (depth_map < combined_depth))
                combined_depth[update_pixels] = depth_map[update_pixels]
        
        return combined_depth
    
    def save_depth_image(self, depth_map, output_path, as_float16=True):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if as_float16:
            # Convert to float16 and view as uint16 for storage
            depth_out = depth_map.astype(np.float16).view(np.uint16)
            cv2.imwrite(output_path, depth_out)
        else:
            # save as raw float32
            cv2.imwrite(output_path, depth_map)
        
        return output_path
    
    def visualize_depth(self, depth, cmap="jet", is_sparse=True, save_path=None):
        if isinstance(depth, str):
            depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
            if depth.dtype == np.uint16:
                depth = depth.view(np.float16).astype(np.float32)
        
        x = np.nan_to_num(depth)
        inv_depth = 1 / (x + 1e-6)

        if is_sparse:
            vmax = 1 / np.percentile(x[x != 0], 5)
        else:
            vmax = np.percentile(inv_depth, 95)

        normalizer = matplotlib.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
        if is_sparse:
            vis_data[inv_depth > vmax] = 0

        if save_path:
            cv2.imwrite(save_path, vis_data)
        else:
            return vis_data
    
    def visualize_rgbd(self, image, depth, cmap="jet", point_size=2, save_path=None):
        v, u = np.nonzero(depth)
        depths = depth[v, u]
        
        plt.figure(figsize=(10, 8))
        if image.shape[2] == 1 or len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        plt.imshow(image)
        sc = plt.scatter(u, v, c=depths, cmap='jet', s=point_size, marker='o', alpha=0.15)
        plt.colorbar(sc, label='Depth (m)')
        plt.title('Projected Depth')

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


def get_lidar2cam_transform(extrinsics):
    T_lidar2cam = np.eye(4)
    p = extrinsics['p']
    q = extrinsics['q']

    T_lidar2cam[:3, 3] = np.array(p)
    T_lidar2cam[:3, :3] = R.from_quat(q).as_matrix()
    return T_lidar2cam

# ========================= Visualization =========================
def visualize_points(points):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def visualize_trajectories(odom_poses, camera_poses, config):
    num_frames = len(odom_poses)
    print(f"Visualizing {num_frames} poses")
    
    if num_frames == 0:
        print("No poses to visualize")
        return
    
    def pose_7d_to_matrix(pose_7d):
        T = np.eye(4)
        T[:3, 3] = pose_7d[:3]
        T[:3, :3] = R.from_quat(pose_7d[3:7]).as_matrix()
        return T
    
    odom_transforms = [pose_7d_to_matrix(pose) for pose in odom_poses]
    camera_transforms = [pose_7d_to_matrix(pose) for pose in camera_poses]
    
    odom_positions = np.array([T[:3, 3] for T in odom_transforms])
    odom_rotations = np.array([T[:3, :3] for T in odom_transforms])
    camera_positions = np.array([T[:3, 3] for T in camera_transforms])
    camera_rotations = np.array([T[:3, :3] for T in camera_transforms])
    
    print(f"Odometry position range: {odom_positions.min(axis=0)} to {odom_positions.max(axis=0)}")
    print(f"Camera position range: {camera_positions.min(axis=0)} to {camera_positions.max(axis=0)}")

    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect([1,1,1])
    ax1.grid(True)
    ax1.set_title('Combined Trajectories', fontsize=14)
    
    odom_line = ax1.plot(odom_positions[:, 0], odom_positions[:, 1], odom_positions[:, 2], 
            label=f'Odometry ({config["odometry_frame"]})', color='blue', linewidth=3)[0]
    camera_line = ax1.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
            label=f'Camera ({config["camera_frame"]})', color='red', linewidth=4)[0]
    
    odom_max_range = max(np.ptp(odom_positions[:, 0]), np.ptp(odom_positions[:, 1]), np.ptp(odom_positions[:, 2]))
    odom_x_mid = (odom_positions[:, 0].max() + odom_positions[:, 0].min()) / 2
    odom_y_mid = (odom_positions[:, 1].max() + odom_positions[:, 1].min()) / 2
    odom_z_mid = (odom_positions[:, 2].max() + odom_positions[:, 2].min()) / 2
    padding = odom_max_range * 0.1
    ax1.set_xlim(odom_x_mid - odom_max_range/2 - padding, odom_x_mid + odom_max_range/2 + padding)
    ax1.set_ylim(odom_y_mid - odom_max_range/2 - padding, odom_y_mid + odom_max_range/2 + padding)
    ax1.set_zlim(odom_z_mid - odom_max_range/2 - padding, odom_z_mid + odom_max_range/2 + padding)
    
    ax1.scatter(odom_positions[0, 0], odom_positions[0, 1], odom_positions[0, 2], 
              color='green', s=150, marker='o', label='Start', edgecolor='black')
    ax1.scatter(odom_positions[-1, 0], odom_positions[-1, 1], odom_positions[-1, 2], 
              color='orange', s=150, marker='s', label='End', edgecolor='black')
    
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect([1,1,1])
    ax2.grid(True)
    ax2.set_title('Camera Trajectory (Zoomed)', fontsize=14)
    
    ax2.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
            label=f'Camera ({config["camera_frame"]})', color='red', linewidth=4)
    
    cam_max_range = max(np.ptp(camera_positions[:, 0]), np.ptp(camera_positions[:, 1]), np.ptp(camera_positions[:, 2]))
    if cam_max_range < 0.01:
        cam_max_range = 0.1
    cam_x_mid = (camera_positions[:, 0].max() + camera_positions[:, 0].min()) / 2
    cam_y_mid = (camera_positions[:, 1].max() + camera_positions[:, 1].min()) / 2
    cam_z_mid = (camera_positions[:, 2].max() + camera_positions[:, 2].min()) / 2
    cam_padding = cam_max_range * 0.2
    ax2.set_xlim(cam_x_mid - cam_max_range/2 - cam_padding, cam_x_mid + cam_max_range/2 + cam_padding)
    ax2.set_ylim(cam_y_mid - cam_max_range/2 - cam_padding, cam_y_mid + cam_max_range/2 + cam_padding)
    ax2.set_zlim(cam_z_mid - cam_max_range/2 - cam_padding, cam_z_mid + cam_max_range/2 + cam_padding)
    
    n = max(1, num_frames // 10)
    arrow_scale = cam_max_range * 0.1
    colors = ['red', 'green', 'blue']
    
    for i in range(0, num_frames, n):
        pos = camera_positions[i]
        rot_matrix = camera_rotations[i]
        for j in range(3):
            ax2.quiver(pos[0], pos[1], pos[2],
                     rot_matrix[0, j] * arrow_scale,
                     rot_matrix[1, j] * arrow_scale, 
                     rot_matrix[2, j] * arrow_scale,
                     color=colors[j], arrow_length_ratio=0.3, alpha=0.8, linewidth=2)
    
    ax2.scatter(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2], 
              color='green', s=150, marker='o', label='Start', edgecolor='black')
    ax2.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], 
              color='orange', s=150, marker='s', label='End', edgecolor='black')
    
    x_axis = ax2.plot([], [], color='red', label='X-axis', linewidth=3, alpha=0.7)[0]
    y_axis = ax2.plot([], [], color='green', label='Y-axis', linewidth=3, alpha=0.7)[0]
    z_axis = ax2.plot([], [], color='blue', label='Z-axis', linewidth=3, alpha=0.7)[0]
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_zlabel('Z (m)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    odom_distance = np.sum(np.linalg.norm(np.diff(odom_positions, axis=0), axis=1))
    camera_distance = np.sum(np.linalg.norm(np.diff(camera_positions, axis=0), axis=1))
    print(f"Odometry distance: {odom_distance:.2f} m, Camera distance: {camera_distance:.2f} m")
