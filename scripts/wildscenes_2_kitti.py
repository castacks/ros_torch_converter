import os
import yaml
import tqdm
import argparse

import cv2
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

from ros_torch_converter.datatypes.image import ImageTorch
from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch

from tartandriver_utils.geometry_utils import pose_to_htm, transform_points

from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *

DIRNAME_3D = "WildScenes3d"
DIRNAME_2D = "WildScenes2d"
DIRNAME_FULLCLOUD = "Fullclouds"

def proc_run_dir(save_dir, pc_pose_fp, pc_dir, img_pose_fp, img_dir):
    """
    Convert WildScenes data into our KITTI format.
    Note that images and pcs arent 1-to-1 correspondence. To address this we'll 
      do a passthrough on images, and for each image, get the closest pointcloud (in odom)
    """
    #setup folders
    odom_outdir = os.path.join(save_dir, 'odom')
    os.makedirs(odom_outdir, exist_ok=True)

    pc_outdir = os.path.join(save_dir, 'super_odometry_pc')
    os.makedirs(pc_outdir, exist_ok=True)

    img_outdir = os.path.join(save_dir, 'image_left_color')
    os.makedirs(img_outdir, exist_ok=True)

    pc_pose_df = pd.read_csv(pc_pose_fp, delimiter=' ', )

    pc_pose_ts = pc_pose_df['timestamp'].to_numpy()
    pc_poses = pc_pose_df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy()

    img_pose_df = pd.read_csv(img_pose_fp, delimiter=' ', )

    img_pose_ts = img_pose_df['timestamp'].to_numpy()
    img_poses = img_pose_df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy()

    """
    viz
    plt.scatter(pc_poses[:, 0], pc_poses[:, 1], c='b', label='pc poses')
    plt.scatter(img_poses[:, 0], img_poses[:, 1], c='r', label='img poses')
    plt.gca().set_aspect(1.)
    plt.legend()
    plt.show()
    """

    img_fps = sorted(os.listdir(img_dir))
    img_ts = np.array([float(fp.split('.')[0].replace('-', '.')) for fp in img_fps])
    pc_fps = sorted(os.listdir(pc_dir))
    pc_ts = np.array([float(fp.split('.')[0]) for fp in pc_fps])

    bad_cnt = 0
    cnt = 0

    out_ts = []

    for img_idx in tqdm.tqdm(range(len(img_fps))):
        img_fp = img_fps[img_idx]
        img_t = img_ts[img_idx]

        pc_tdiffs = np.abs(pc_ts - img_t)
        pc_idx = np.argmin(pc_tdiffs)

        pc_fp = pc_fps[pc_idx]
        pc_t = pc_ts[pc_idx]

        if pc_tdiffs[pc_idx] > 1.0:
            # print('warning: pc_offset > 1.0s. skipping...')
            bad_cnt += 1
            continue

        img = cv2.imread(os.path.join(img_dir, img_fp)) / 255.
        pc = o3d.io.read_point_cloud(os.path.join(pc_dir, pc_fp))
        pose = np.zeros(13)
        pose[:7] = img_poses[img_idx]

        pc_pose = pc_poses[pc_idx]
        pc_htm = pose_to_htm(pc_pose)
        pc.transform(pc_htm)

        img_torch = ImageTorch.from_numpy(img, device='cpu')
        pc_torch = PointCloudTorch.from_numpy(np.asarray(pc.points), device='cpu')
        pose_torch = OdomRBStateTorch.from_numpy(pose, child_frame_id="base_link", device='cpu')

        # import pdb;pdb.set_trace()

        img_torch.stamp = img_t
        pc_torch.stamp = img_t
        pose_torch.stamp = img_t

        img_torch.to_kitti(base_dir=img_outdir, idx=cnt)
        pc_torch.to_kitti(base_dir=pc_outdir, idx=cnt)
        pose_torch.to_kitti(base_dir=odom_outdir, idx=cnt)

        #debug colorization

        # if (img_idx % 100000000) == 0:
        #     lidar_to_cam = np.array([
        #         # 0.0694452427858,-0.00671311927669,0.0208717486986,
        #         #  0.00274514, -0.13766556,  0.00158476,  0.9904737 
        #         0., 0., 0.,
        #         0., 0., 0., 1.
        #     ])
        #     extrinsics = torch.tensor(pose_to_htm(lidar_to_cam)).float()

        #     intrinsics = get_intrinsics(torch.tensor(
        #         [1322.75469666, 0., 1014.8117275, 0., 1321.88964261, 752.801443314, 0., 0., 1.]
        #     ).reshape(3, 3)).float()

        #     P = obtain_projection_matrix(intrinsics, extrinsics)
        #     img_pose_htm = torch.tensor(pose_to_htm(img_poses[img_idx])).float()
        #     pcl = transform_points(pc_torch.pts, torch.linalg.inv(img_pose_htm))
        #     pcl_dists = torch.linalg.norm(pcl, dim=-1)

        #     pcl_pixel_coords, ind_in_frame = get_pixel_from_3D_source(pcl, P, img)

        #     # (
        #     #     pcl_in_frame,
        #     #     pixels_in_frame,
        #     #     ind_in_frame,
        #     # ) = get_points_and_pixels_in_frame(
        #     #     pcl, pcl_pixel_coords, img.shape[0], img.shape[1]
        #     # )

        #     pcl_px_in_frame = pcl_pixel_coords[ind_in_frame]
        #     pcl_dists = pcl_dists[ind_in_frame]
        #     pcl_dists = np.clip(pcl_dists, 2., 20.)
        #     pc_z = (pcl_dists / pcl_dists.max()).cpu().numpy()

        #     fig, axs = plt.subplots(1, 2, figsize=(40, 24))
        #     axs = axs.flatten()

        #     axs[0].imshow(img)
        #     axs[1].imshow(img)
        #     axs[1].scatter(pcl_px_in_frame[:, 0].cpu(), pcl_px_in_frame[:, 1].cpu(), c=pc_z, s=1., alpha=0.5, cmap='jet')

        #     plt.show()

        out_ts.append(img_t)

        cnt += 1

    #save timestamps
    out_ts = np.array(out_ts)
    np.savetxt(os.path.join(img_outdir, 'timestamps.txt'), out_ts)
    np.savetxt(os.path.join(pc_outdir, 'timestamps.txt'), out_ts)
    np.savetxt(os.path.join(odom_outdir, 'timestamps.txt'), out_ts)

    print('{} good samples'.format(cnt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='location of wildscenes root dir')
    parser.add_argument('--save_dir', type=str, required=True, help='location to save kitti dataset')
    args = parser.parse_args()

    wildscenes_3d_dir = os.path.join(args.root_dir, DIRNAME_3D)
    wildscenes_2d_dir = os.path.join(args.root_dir, DIRNAME_2D)
    wildscenes_fullcloud_dir = os.path.join(args.root_dir, DIRNAME_FULLCLOUD)

    run_dirs = os.listdir(wildscenes_3d_dir)

    print("processing these runs: {}".format(run_dirs))

    for rdir in run_dirs:
        pc_pose_fp = os.path.join(wildscenes_3d_dir, rdir, "poses3d.csv")
        pc_dir = os.path.join(wildscenes_fullcloud_dir, rdir)
        img_pose_fp = os.path.join(wildscenes_2d_dir, rdir, 'poses2d.csv')
        img_dir = os.path.join(wildscenes_2d_dir, rdir, "image")

        save_dir = os.path.join(args.save_dir, rdir)

        proc_run_dir(save_dir, pc_pose_fp, pc_dir, img_pose_fp, img_dir)