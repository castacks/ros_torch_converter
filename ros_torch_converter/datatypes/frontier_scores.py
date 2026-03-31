import os
import yaml
import h5py
import warnings

import torch
import numpy as np

from geometry_msgs.msg import Point

from perception_interfaces.msg import FrontierScores

from tartandriver_utils.os_utils import save_yaml
from tartandriver_utils.ros_utils import time_to_stamp

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

class FrontierScoresTorch(TorchCoordinatorDataType): 
    """
    TorchCoordinator class for long-range frontier scores.
    Essentially, an array of headings + scores per headings
    """
    to_rosmsg_type = FrontierScores
    from_rosmsg_type = FrontierScores
    time_spec = TimeSpec.SYNC

    def __init__(self, device):
        super().__init__()
        self.binwidth = -1.
        self.scores = torch.zeros(0, device=device)
        self.headings = torch.zeros(0, device=device)
        self.position = torch.zeros(3, device=device)

        self.device = device

    def from_torch(scores, headings, binwidth, position):
        assert binwidth > 0.

        res = FrontierScoresTorch(device=scores.device)

        res.binwidth = binwidth
        res.scores = scores
        res.headings = headings
        res.position = position
        
        return res

    def from_numpy(scores, headings, binwidth, position, device='cpu'):
        res = FrontierScoresTorch(device=device)

        res.binwidth = binwidth
        res.scores = torch.tensor(scores, dtype=torch.float32, device=device)
        res.headings = torch.tensor(headings, dtype=torch.float32, device=device)
        res.position = torch.tensor(position, dtype=torch.float32, device=device)

        return res
    
    def from_rosmsg(msg, device='cpu'):
        res = FrontierScoresTorch.from_numpy(
            scores = np.array(msg.scores),
            headings = np.array(msg.headings),
            binwidth = msg.binwidth,
            position = np.array([
                msg.position.x,
                msg.position.y,
                msg.position.z,
            ]),
            device = device
        )
        res.stamp = msg.header.stamp
        res.frame_id = msg.header.frame_id
        return res

    def to_rosmsg(self):
        msg = FrontierScores()

        msg.binwidth = self.binwidth
        msg.headings = self.headings.cpu().numpy().tolist()
        msg.scores = self.scores.cpu().numpy().tolist()
        msg.position = Point(
            x = self.position[0].item(),
            y = self.position[1].item(),
            z = self.position[2].item()
        )
        
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg

    def to_kitti(self, base_dir, idx):
        """define how to convert this dtype to a kitti file
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_info_file(base_dir, 'frame_id', self.frame_id)

        data = {
            'headings': self.headings.cpu().numpy(),
            'scores': self.scores.cpu().numpy(),
            'position': self.position.cpu().numpy(),
            'binwidth': self.binwidth
        }

        data_fp = os.path.join(base_dir, "{:08d}_data.npz".format(idx))
        np.savez(data_fp, **data)

    def from_kitti(base_dir, idx, device='cpu'):
        """define how to convert this dtype from a kitti file
        """
        data_fp = os.path.join(base_dir, "{:08d}_data.npz".format(idx))
        data = np.load(data_fp)

        res = FrontierScoresTorch.from_numpy(
            binwidth = data['binwidth'].item(),
            headings = data['headings'],
            scores = data['scores'],
            position = data['position'],
            device = device
        )

        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_info_file(base_dir,  'frame_id')

        return res

    def to(self, device):
        self.device = device
        self.headings = self.headings.to(device)
        self.scores = self.scores.to(device)
        self.position = self.position.to(device)
        return self
    
    def rand_init(device='cpu'):
        N = np.random.randint(5, 100)
        rad_range = np.random.rand() * 2 * np.pi
        bw = rad_range / N

        scores = np.random.rand(N).reshape(-1)
        headings = np.linspace(0., 1., N) * (2 * np.pi)

        position = np.random.rand(3)

        out = FrontierScoresTorch.from_numpy(
            binwidth = bw,
            scores = scores,
            headings = headings,
            position = position,
            device = device
        )

        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if abs(self.binwidth - other.binwidth) > 1e-8:
            return False

        if not torch.allclose(self.headings, other.headings):
            return False
        
        if not torch.allclose(self.scores, other.scores):
            return False
        
        if not torch.allclose(self.position, other.position):
            return False

        return True        

    def __repr__(self):
        return "FrontierScoresTorch from {:.2f}-{:.2f} ({} bins) (time = {:.2f}, frame = {}, device = {})".format(self.headings.min(), self.headings.max(), self.headings.shape[0], self.stamp, self.frame_id, self.device)
