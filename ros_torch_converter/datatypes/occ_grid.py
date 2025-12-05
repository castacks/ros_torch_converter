import torch

import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType


from nav_msgs.msg import OccupancyGrid

from tartandriver_utils.ros_utils import time_to_stamp


class OccupancyGridTorch(TorchCoordinatorDataType):
    """
    Wrapper around the BEVGrid class from visual mapping
    """
    to_rosmsg_type = OccupancyGrid
    from_rosmsg_type = OccupancyGrid

    def __init__(self, device):
        super().__init__()
        self.bev_grid = None
        self.height = torch.tensor(0., device=device)
        self.device = device
    
    def from_rosmsg(msg, feature_keys=[], device='cpu'):
        pass

    def to_rosmsg(self):
        occgrid_msg = OccupancyGrid()
        
        occgrid_msg.header.stamp = time_to_stamp(self.stamp)
        occgrid_msg.header.frame_id = self.frame_id

        occ_idx = self.bev_grid.feature_keys.index("occupancy")
        
        # Handle NaNs which can occur when the map is empty/uninitialized
        data = self.bev_grid.data[..., occ_idx].T.flatten().cpu().numpy()
        data = np.nan_to_num(data, nan=-1.0)
        occgrid_data = data.astype(int).tolist()

        occgrid_msg.info.resolution = self.bev_grid.metadata.resolution[0].item()
        occgrid_msg.info.width = self.bev_grid.metadata.N[0].item()
        occgrid_msg.info.height = self.bev_grid.metadata.N[1].item()
        occgrid_msg.info.origin.position.x = self.bev_grid.metadata.origin[0].item()
        occgrid_msg.info.origin.position.y = self.bev_grid.metadata.origin[1].item()
        occgrid_msg.info.origin.position.z = self.height.item()
        occgrid_msg.info.origin.orientation.w = 1.0

        occgrid_msg.data = occgrid_data

        return occgrid_msg

    def to_kitti(self, base_dir, idx):
        pass

    def from_kitti(base_dir, idx, device='cpu'):
        pass

    def to(self, device):
        self.device = device
        self.bev_grid = self.bev_grid.to(device)
        return self
    
    def __repr__(self):
        return "BEVGridTorch of size {}, time = {:.2f}, frame = {}, device = {}".format(self.bev_grid.data.shape, self.stamp, self.frame_id, self.device)