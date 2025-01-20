import copy
import warnings
import torch
import numpy as np
import ros2_numpy

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.utils import normalize_dino

from sensor_msgs.msg import PointCloud2

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
        pass
        return None

    def to_rosmsg(self):
        """
        For now, colors are a scaling of the first 3 features
        """
        feature_idxs = self.voxel_grid.indices
        non_feature_idxs = self.voxel_grid.non_feature_indices

        all_idxs = torch.cat([feature_idxs, non_feature_idxs])
        all_pts = self.voxel_grid.grid_indices_to_pts(self.voxel_grid.raster_indices_to_grid_indices(all_idxs))

        feature_colors = normalize_dino(self.voxel_grid.features[:, :3])
        non_feature_colors = 0.8 * torch.ones(non_feature_idxs.shape[0], 3, device=self.device)
        all_colors = torch.cat([feature_colors, non_feature_colors], dim=0)

        points = all_pts.cpu().numpy()
        rgb_values = (all_colors * 255.0).cpu().numpy().astype(np.uint8)
        # Prepare the data array with XYZ and RGB
        xyzcolor = np.zeros(
            points.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
            ],
        )

        # Assign XYZ values
        xyzcolor["x"] = points[:, 0]
        xyzcolor["y"] = points[:, 1]
        xyzcolor["z"] = points[:, 2]

        color = np.zeros(
            points.shape[0], dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8)]
        )
        color["r"] = rgb_values[:, 0]
        color["g"] = rgb_values[:, 1]
        color["b"] = rgb_values[:, 2]
        xyzcolor["rgb"] = ros2_numpy.point_cloud2.merge_rgb_fields(color)

        msg = ros2_numpy.msgify(PointCloud2, xyzcolor)

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
        return "VoxelGridTorch of size {}, time = {:.2f}, frame = {}, device = {}".format(self.voxel_grid.features.shape, self.stamp, self.frame_id, self.device)