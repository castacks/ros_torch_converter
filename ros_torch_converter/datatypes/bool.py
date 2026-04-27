import os
import torch
import numpy as np

from tartandriver_utils.geometry_utils import MultiDimensionalInterpolator
from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

from core_interfaces.msg import BoolStamped

class BoolTorch(TorchCoordinatorDataType):
    """
    """
    to_rosmsg_type = BoolStamped
    from_rosmsg_type = BoolStamped
    time_spec = TimeSpec.SYNC # interpolating bools is nonsense

    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        self.data = torch.zeros(1, dtype=bool, device=device)
        self.device = device
    
    def from_torch(x):
        out = BoolTorch(device=x.device)
        out.data = x
        return out

    def from_rosmsg(msg, device='cpu'):
        res = BoolTorch(device=device)
        print(msg.data)
        res.data = torch.tensor([msg.data], device=device)
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res
    
    def to_rosmsg(self):
        msg = BoolStamped()
        msg.data = self.data.item()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg
    
    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        update_timestamp_file(base_dir, idx, self.stamp, file='timestamps.txt')
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='data.txt')

    def save_to_file(self, base_dir, idx, file='data.txt'):
        save_fp = os.path.join(base_dir, file)
        if not os.path.exists(save_fp):
            data = float('inf') * np.ones([idx+1])
        else:
            #need to reshape for 1-row data
            data = np.loadtxt(save_fp).reshape(-1)

        if data.shape[0] < (idx+1):
            data_new = float('inf') * np.ones([idx+1])
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = self.data.cpu().numpy()

        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(fp).reshape(-1)[idx]

        out = BoolTorch(device=device)
        out.data = torch.tensor(data, device=device).bool()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_info_file(base_dir,  'frame_id')

        return out
    
    def from_kitti_multi(base_dir, idxs, device='cpu'):
        fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(fp).reshape(-1)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_ids = read_info_file(base_dir,'frame_id')

        out = [BoolTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = BoolTorch(device=device)
        out.data = torch.randint(0,2,size=(), device=device, dtype=bool)
        out.frame_id = 'random'
        out.stamp = np.random.rand()

        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.data, other.data):
            return False

        return True

    def __repr__(self):
        return "BoolTorch with data {}, device {}".format(self.data.item(), self.device)