import copy
import array
import torch
import numpy as np
import ros2_numpy

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.utils import normalize_dino

from sensor_msgs.msg import PointCloud2, PointField

from tartandriver_utils.ros_utils import time_to_stamp, stamp_to_time

class VoxelGridTorch(TorchCoordinatorDataType):
    """
    Wrapper around the VoxelGrid class from visual mapping
    """
    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, device):
        super().__init__()
        self.voxel_grid = None
        self.device = device

    def from_voxel_grid(voxel_grid):
        res = VoxelGridTorch(device=voxel_grid.device)
        res.voxel_grid = voxel_grid
        return res
    
    def from_rosmsg(msg, feature_keys=[], device='cpu'):
        return None

    def to_rosmsg(self):
        """
        For now, colors are a scaling of the first 3 features
        """
        feature_idxs = self.voxel_grid.feature_raster_indices
        non_feature_idxs = self.voxel_grid.non_feature_raster_indices

        all_idxs = torch.cat([feature_idxs, non_feature_idxs])
        all_pts = self.voxel_grid.grid_indices_to_pts(self.voxel_grid.raster_indices_to_grid_indices(all_idxs))
        points = all_pts.cpu().numpy().astype(np.float32)

        msg = PointCloud2()
        msg.height = 1
        msg.width = points.shape[0]
        msg.point_step = 12

        msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=msg.width) for i,n in enumerate('xyz')]

        data = points

        if self.voxel_grid.features.shape[1] >= 3:
            feature_colors = normalize_dino(self.voxel_grid.features[:, :3])
            non_feature_colors = 0.8 * torch.ones(non_feature_idxs.shape[0], 3, device=self.device)
            all_colors = torch.cat([feature_colors, non_feature_colors], dim=0)
            colors = all_colors.cpu().numpy()

            msg.point_step += 4
            msg.fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=msg.width))

            r = (colors[:, 0] * 255).astype(np.uint32)
            g = (colors[:, 1] * 255).astype(np.uint32)
            b = (colors[:, 2] * 255).astype(np.uint32)
            rgb = (r<<16) | (g<<8) | (b<<0)
            rgb.dtype = np.float32

            data = np.concatenate([data, rgb.reshape(-1, 1)], axis=-1)

        data = data.flatten()

        # borrowing from https://github.com/Box-Robotics/ros2_numpy/blob/humble/ros2_numpy/point_cloud2.py
        mem_view = memoryview(data)

        if mem_view.nbytes > 0:
            array_bytes = mem_view.cast("B")
        else:
            array_bytes = b""

        as_array = array.array("B")
        as_array.frombytes(array_bytes)

        msg.data = as_array
            
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg

    def to_kitti(self, base_dir, idx):
        """define how to convert this dtype to a kitti file
        """
        pass

    def from_kitti(self, base_dir, idx, device):
        """define how to convert this dtype from a kitti file
        """
        pass

    def to(self, device):
        self.device = device
        self.voxel_grid = self.voxel_grid.to(device)
        return self
    
    def __repr__(self):
        return "VoxelGridTorch of size {}, {}, time = {:.2f}, frame = {}, device = {}".format(self.voxel_grid.features.shape, self.voxel_grid.raster_indices.shape, self.stamp, self.frame_id, self.device)