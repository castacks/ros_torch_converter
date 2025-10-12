import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import (
    update_frame_file, update_timestamp_file,
    read_frame_file, read_timestamp_file
)

from std_msgs.msg import Bool


class BoolTorch(TorchCoordinatorDataType):
    """
    TorchCoordinator wrapper for std_msgs/Bool
    """
    to_rosmsg_type = Bool
    from_rosmsg_type = Bool

    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        # store as torch.bool tensor
        self.data = torch.zeros(1, dtype=torch.bool, device=device)
        self.device = device

    @staticmethod
    def from_rosmsg(msg, device='cpu'):
        res = BoolTorch(device=device)
        res.data = torch.tensor([msg.data], dtype=torch.bool, device=device)
        return res

    def to_rosmsg(self):
        msg = Bool()
        msg.data = bool(self.data.item())
        return msg

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        Store boolean values as integers (0/1) in KITTI-style txt
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "data.txt")
        if not os.path.exists(save_fp):
            data = np.full([idx + 1], fill_value=np.inf)
        else:
            data = np.loadtxt(save_fp).reshape(-1)

        if data.shape[0] < (idx + 1):
            data_new = np.full([idx + 1], fill_value=np.inf)
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = int(self.data.item())

        np.savetxt(save_fp, data, fmt='%d')

    @staticmethod
    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "data.txt")
        data = np.loadtxt(fp).reshape(-1)[idx]

        out = BoolTorch(device=device)
        out.data = torch.tensor(bool(data), dtype=torch.bool, device=device)

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return out

    @staticmethod
    def rand_init(device='cpu'):
        out = BoolTorch(device=device)
        out.data = torch.randint(0, 2, (1,), dtype=torch.bool, device=device)
        out.frame_id = 'random'
        out.stamp = np.random.rand()
        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if self.data.item() != other.data.item():
            return False
        return True

    def __repr__(self):
        return f"BoolTorch with data {self.data.item()}, device {self.device}"
