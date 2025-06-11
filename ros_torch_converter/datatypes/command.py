import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from geometry_msgs.msg import TwistStamped

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class CommandTorch(TorchCoordinatorDataType):
    """command as [vx, wz]
    """
    to_rosmsg_type = TwistStamped
    from_rosmsg_type = TwistStamped
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.state = torch.zeros(2, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = CommandTorch(device=device)

        res.state = torch.tensor([
            msg.twist.linear.x,
            msg.twist.angular.z
        ], device=device).float()
        
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(data, device='cpu'):
        return CommandTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = CommandTorch(device=data.device)
        res.state = data

        return res

    def to_rosmsg(self):
        msg = TwistStamped()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id

        msg.twist.linear.x = self.state[0].item()
        msg.twist.angular.z = self.state[1].item()

        return msg

    def to(self, device):
        self.device = device
        self.state = self.state.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        save_fp = os.path.join(base_dir, "data.txt")
        if not os.path.exists(save_fp):
            data = float('inf') * np.ones([idx+1, 2])
        else:
            #need to reshape for 1-row data
            data = np.loadtxt(save_fp).reshape(-1, 2)

        if data.shape[0] < (idx+1):
            data_new = float('inf') * np.ones([idx+1, 2])
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = self.state.cpu().numpy()

        np.savetxt(save_fp, data)

    def from_kitti(self, base_dir, idx, device='cpu'):
        pass

    def __repr__(self):
        return "CommandTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.state.cpu().numpy().round(4), self.stamp, self.device)
    