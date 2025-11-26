import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import update_frame_file, update_timestamp_file, read_frame_file, read_timestamp_file

from racepak_interfaces.msg import RpControls, RpShockSensors, RpWheelEncoders

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class RacepakPedalPosTorch(TorchCoordinatorDataType):
    """Pedal position as [throttle, brake]
    """
    to_rosmsg_type = RpControls
    from_rosmsg_type = RpControls
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.data = torch.zeros(2, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = RacepakPedalPosTorch(device=device)

        res.data = torch.tensor([
            msg.throttle,
            msg.brake
        ], device=device).float()
        
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(data, device='cpu'):
        return RacepakPedalPosTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = RacepakPedalPosTorch(device=data.device)
        res.data = data

        return res

    def to_rosmsg(self):
        msg = RpControls()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id

        msg.throttle = self.data[0].item()
        msg.brake = self.data[1].item()

        return msg

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

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

        data[idx] = self.data.cpu().numpy()

        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 2)[idx]

        out = RacepakPedalPosTorch(device=device)
        out.data = torch.tensor(data, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return out
    
    def from_kitti_multi(base_dir, idxs, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 2)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_frame_file(base_dir, idxs[0], 'frame_id')

        out = [RacepakPedalPosTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = RacepakPedalPosTorch(device)
        out.data = torch.rand(2, device=device)
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
        return "RacepakPedalPosTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.data.cpu().numpy().round(4), self.stamp, self.device)
    
class RacepakShockPosTorch(TorchCoordinatorDataType):
    """shock position as [d_FL, d_FR, d_RL, d_RR]
    """
    to_rosmsg_type = RpShockSensors
    from_rosmsg_type = c
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.data = torch.zeros(4, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = RacepakShockPosTorch(device=device)

        res.data = torch.tensor([
            msg.front_left,
            msg.front_right,
            msg.rear_left,
            msg.rear_right
        ], device=device).float()
        
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(data, device='cpu'):
        return RacepakShockPosTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = RacepakShockPosTorch(device=data.device)
        res.data = data

        return res

    def to_rosmsg(self):
        msg = RpWheelEncoders()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id

        msg.front_left = self.data[0].item()
        msg.front_right = self.data[1].item()
        msg.rear_left = self.data[2].item()
        msg.rear_right = self.data[3].item()

        return msg

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "data.txt")
        if not os.path.exists(save_fp):
            data = float('inf') * np.ones([idx+1, 4])
        else:
            #need to reshape for 1-row data
            data = np.loadtxt(save_fp).reshape(-1, 4)

        if data.shape[0] < (idx+1):
            data_new = float('inf') * np.ones([idx+1, 4])
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = self.data.cpu().numpy()

        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 4)[idx]

        out = RacepakShockPosTorch(device=device)
        out.data = torch.tensor(data, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return out
    
    def from_kitti_multi(base_dir, idxs, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 4)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_frame_file(base_dir, idxs[0], 'frame_id')

        out = [RacepakShockPosTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = RacepakShockPosTorch(device)
        out.data = torch.rand(4, device=device)
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
        return "RacepakShockPosTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.data.cpu().numpy().round(4), self.stamp, self.device)
    
class RacepakWheelRPMTorch(TorchCoordinatorDataType):
    """wheel RPM as [w_FL, w_FR, w_RL, w_RR]
    """
    to_rosmsg_type = RpWheelEncoders
    from_rosmsg_type = RpWheelEncoders
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.data = torch.zeros(4, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = RacepakWheelRPMTorch(device=device)

        res.data = torch.tensor([
            msg.front_left,
            msg.front_right,
            msg.rear_left,
            msg.rear_right
        ], device=device).float()
        
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(data, device='cpu'):
        return RacepakWheelRPMTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = RacepakWheelRPMTorch(device=data.device)
        res.data = data

        return res

    def to_rosmsg(self):
        msg = RpWheelEncoders()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id

        msg.front_left = self.data[0].item()
        msg.front_right = self.data[1].item()
        msg.rear_left = self.data[2].item()
        msg.rear_right = self.data[3].item()

        return msg

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "data.txt")
        if not os.path.exists(save_fp):
            data = float('inf') * np.ones([idx+1, 4])
        else:
            #need to reshape for 1-row data
            data = np.loadtxt(save_fp).reshape(-1, 4)

        if data.shape[0] < (idx+1):
            data_new = float('inf') * np.ones([idx+1, 4])
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = self.data.cpu().numpy()

        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 4)[idx]

        out = RacepakWheelRPMTorch(device=device)
        out.data = torch.tensor(data, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return out
    
    def from_kitti_multi(base_dir, idxs, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 4)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_frame_file(base_dir, idxs[0], 'frame_id')

        out = [RacepakWheelRPMTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = RacepakWheelRPMTorch(device)
        out.data = torch.rand(4, device=device)
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
        return "RacepakWheelRPMTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.data.cpu().numpy().round(4), self.stamp, self.device)
    