import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from std_msgs.msg import Float32

class Float32Torch(TorchCoordinatorDataType):
    """
    """
    to_rosmsg_type = Float32
    from_rosmsg_type = Float32

    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        self.data = torch.zeros(1, device=device)
        self.device = device
    
    def from_rosmsg(msg, device='cpu'):
        res = Float32Torch(device=device)
        res.data = torch.tensor([msg.data], device=device)
        return res
    
    def to_rosmsg(self):
        msg = Float32()
        msg.data = self.data.item()
        return msg
    
    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        save_fp = os.path.join(base_dir, "data.txt")
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
        timestamp_fp = os.path.join(base_dir, "timestamps.txt")

        data = np.loadtxt(fp)[idx]
        ts = np.loadtxt(timestamp_fp)[idx]

        out = Float32Torch(device=device)
        out.data = torch.tensor(data, device=device).float()
        out.stamp = ts

        return out

    def __repr__(self):
        return "Float32Torch with data {:.2f}, device {}".format(self.data.item(), self.device)