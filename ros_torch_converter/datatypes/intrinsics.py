import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from sensor_msgs.msg import CameraInfo

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class IntrinsicsTorch(TorchCoordinatorDataType):
    """
    Class for camera info. This consists of a 3x3 intrinsics matrix
    """
    to_rosmsg_type = CameraInfo
    from_rosmsg_type = CameraInfo

    def __init__(self, device='cpu'):
        super().__init__()
        self.intrinsics = torch.zeros(3, 3, device=device)
        self.device = device
    
    def from_rosmsg(msg, use_p=True, device='cpu'):
        res = IntrinsicsTorch(device=device)
        if use_p:
            res.intrinsics = torch.tensor(msg.p, device=device).reshape(3, 4)[:3, :3]
        else:
            res.intrinsics = torch.tensor(msg.k, device=device).reshape(3, 3)
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_torch(intrinsics):
        res = IntrinsicsTorch(device=intrinsics.device)
        res.intrinsics = intrinsics.float()
        return res

    def from_numpy(intrinsics, device):
        res = IntrinsicsTorch(device=device)
        res.intrinsics = torch.from_numpy(intrinsics, device=device, dtype=torch.float32)
        return res

    def to_rosmsg(self):
        msg = CameraInfo()

        msg.k = self.intrinsics.cpu().numpy().flatten().tolist()
        msg.r = np.eye(3).flatten().tolist()
        P = np.zeros(3, 4); P[:3, :3] = self.intrinsics.cpu().numpy()
        msg.p = P.flatten().tolist()

        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        return msg

    def to_kitti(self, base_dir, idx):
        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.intrinsics.cpu().numpy().flatten())

    def from_kitti(self, base_dir, idx, device='cpu'):
        pass

    def to(self, device):
        self.device = device
        self.intrinsics = self.intrinsics.to(device)
        return self
    

    def __repr__(self):
        return "IntrinsicsTorch with k:\n{}, device = {}".format(self.intrinsics.cpu().numpy().round(4), self.device)