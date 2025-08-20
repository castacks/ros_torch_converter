import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import update_frame_file, update_timestamp_file, read_frame_file, read_timestamp_file

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
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.intrinsics.cpu().numpy().flatten())

    def from_kitti(base_dir, idx, device='cpu'):
        res = IntrinsicsTorch(device=device)

        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp).reshape(3, 3)

        res.intrinsics = torch.tensor(data).float().to(device)
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return res

    def to(self, device):
        self.device = device
        self.intrinsics = self.intrinsics.to(device)
        return self
    
    def rand_init(device='cpu'):
        data = torch.eye(3, device=device)
        data[[0, 1, 0, 1], [0, 1, 2, 2]] = torch.rand(size=(4, )) * 100.

        out = IntrinsicsTorch.from_torch(data)
        out.frame_id = 'random'
        out.stamp = np.random.rand()

        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.intrinsics, other.intrinsics):
            return False

        return True

    def __repr__(self):
        return "IntrinsicsTorch with k:\n{}, device = {}".format(self.intrinsics.cpu().numpy().round(4), self.device)