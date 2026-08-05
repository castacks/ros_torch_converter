import os
import torch
import numpy as np

from tartandriver_utils.geometry_utils import MultiDimensionalInterpolator

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

from racepak_interfaces.msg import RpControls, RpShockSensors, RpWheelEncoders

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class PedalPosTorch(TorchCoordinatorDataType):
    """Pedal position as [throttle, brake]
    """
    to_rosmsg_type = RpControls
    from_rosmsg_type = RpControls
    time_spec = TimeSpec.INTERP
    
    def to_interp(base_dir, datalist):
        data = torch.stack([x.data for x in datalist], dim=0).cpu().numpy()
        times = np.array([x.stamp for x in datalist])

        data_fp = os.path.join(base_dir, 'interp_data.txt')
        timestamp_fp = os.path.join(base_dir, 'interp_timestamps.txt')

        np.savetxt(data_fp, data)
        np.savetxt(timestamp_fp, times)

    def from_interp(base_dir, target_timestamp, device, tol=0.5):
        data_fp = os.path.join(base_dir, 'interp_data.txt')
        timestamp_fp = os.path.join(base_dir, 'interp_timestamps.txt')

        data = np.loadtxt(data_fp).reshape(-1, 2)
        timestamps = np.loadtxt(timestamp_fp)

        interp = MultiDimensionalInterpolator(traj=data, times=timestamps, tol=tol)
        data = interp(target_timestamp)

        out = PedalPosTorch.from_numpy(data).to(device)
        out.stamp = target_timestamp
        out.frame_id = read_info_file(base_dir, 'frame_id')

        return out

    def __init__(self, device='cpu'):
        super().__init__()
        self.data = torch.zeros(2, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = PedalPosTorch(device=device)

        res.data = torch.tensor([
            msg.throttle,
            msg.brake
        ], device=device).float()

        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(data, device='cpu'):
        return PedalPosTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = PedalPosTorch(device=data.device)
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
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='data.txt')

    def to_kitti_interp(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp, file='interp_timestamps.txt')
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='interp_data.txt')

    def save_to_file(self, base_dir, idx, file='data.txt'):
        save_fp = os.path.join(base_dir, file)
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

        out = PedalPosTorch(device=device)
        out.data = torch.tensor(data, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_info_file(base_dir, 'frame_id')

        return out

    def from_kitti_multi(base_dir, idxs, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 2)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_info_file(base_dir, 'frame_id')

        out = [PedalPosTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = PedalPosTorch(device)
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
        return "PedalPosTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.data.cpu().numpy().round(4), self.stamp, self.device)


class ShockPosTorch(TorchCoordinatorDataType):
    """shock position as [d_FL, d_FR, d_RL, d_RR]
    """
    to_rosmsg_type = RpShockSensors
    from_rosmsg_type = RpShockSensors
    time_spec = TimeSpec.INTERP
    
    def to_interp(base_dir, datalist):
        data = torch.stack([x.data for x in datalist], dim=0).cpu().numpy()
        times = np.array([x.stamp for x in datalist])

        data_fp = os.path.join(base_dir, 'interp_data.txt')
        timestamp_fp = os.path.join(base_dir, 'interp_timestamps.txt')

        np.savetxt(data_fp, data)
        np.savetxt(timestamp_fp, times)

    def from_interp(base_dir, target_timestamp, device, tol=0.5):
        data_fp = os.path.join(base_dir, 'interp_data.txt')
        timestamp_fp = os.path.join(base_dir, 'interp_timestamps.txt')

        data = np.loadtxt(data_fp).reshape(-1, 4)
        timestamps = np.loadtxt(timestamp_fp)

        interp = MultiDimensionalInterpolator(traj=data, times=timestamps, tol=tol)
        data = interp(target_timestamp)

        out = ShockPosTorch.from_numpy(data).to(device)
        out.stamp = target_timestamp
        out.frame_id = read_info_file(base_dir, 'frame_id')

        return out

    def __init__(self, device='cpu'):
        super().__init__()
        self.data = torch.zeros(4, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = ShockPosTorch(device=device)

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
        return ShockPosTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = ShockPosTorch(device=data.device)
        res.data = data

        return res

    def to_rosmsg(self):
        msg = RpShockSensors()
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
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='data.txt')

    def to_kitti_interp(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp, file='interp_timestamps.txt')
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='interp_data.txt')

    def save_to_file(self, base_dir, idx, file='data.txt'):
        save_fp = os.path.join(base_dir, file)
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

        out = ShockPosTorch(device=device)
        out.data = torch.tensor(data, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_info_file(base_dir, 'frame_id')

        return out

    def from_kitti_multi(base_dir, idxs, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 4)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_info_file(base_dir, 'frame_id')

        out = [ShockPosTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = ShockPosTorch(device)
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
        return "ShockPosTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.data.cpu().numpy().round(4), self.stamp, self.device)


class WheelRPMTorch(TorchCoordinatorDataType):
    """wheel RPM as [w_FL, w_FR, w_RL, w_RR]
    """
    to_rosmsg_type = RpWheelEncoders
    from_rosmsg_type = RpWheelEncoders
    time_spec = TimeSpec.INTERP
    
    def to_interp(base_dir, datalist):
        data = torch.stack([x.data for x in datalist], dim=0).cpu().numpy()
        times = np.array([x.stamp for x in datalist])

        data_fp = os.path.join(base_dir, 'interp_data.txt')
        timestamp_fp = os.path.join(base_dir, 'interp_timestamps.txt')

        np.savetxt(data_fp, data)
        np.savetxt(timestamp_fp, times)

    def from_interp(base_dir, target_timestamp, device, tol=0.5):
        data_fp = os.path.join(base_dir, 'interp_data.txt')
        timestamp_fp = os.path.join(base_dir, 'interp_timestamps.txt')

        data = np.loadtxt(data_fp).reshape(-1, 4)
        timestamps = np.loadtxt(timestamp_fp)

        interp = MultiDimensionalInterpolator(traj=data, times=timestamps, tol=tol)
        data = interp(target_timestamp)

        out = WheelRPMTorch.from_numpy(data).to(device)
        out.stamp = target_timestamp
        out.frame_id = read_info_file(base_dir, 'frame_id')

        return out

    def __init__(self, device='cpu'):
        super().__init__()
        self.data = torch.zeros(4, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = WheelRPMTorch(device=device)

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
        return WheelRPMTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device))

    def from_torch(data,):
        res = WheelRPMTorch(device=data.device)
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
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='data.txt')

    def to_kitti_interp(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp, file='interp_timestamps.txt')
        update_info_file(base_dir, 'frame_id', self.frame_id)
        self.save_to_file(base_dir, idx, file='interp_data.txt')

    def save_to_file(self, base_dir, idx, file='data.txt'):
        save_fp = os.path.join(base_dir, file)
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

        out = WheelRPMTorch(device=device)
        out.data = torch.tensor(data, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_info_file(base_dir, 'frame_id')

        return out

    def from_kitti_multi(base_dir, idxs, device='cpu'):
        save_fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(save_fp).reshape(-1, 4)[idxs]
        data = torch.tensor(data, device=device).float()
        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_info_file(base_dir, 'frame_id')

        out = [WheelRPMTorch.from_kitti(x) for x in data]

        return out

    def rand_init(device='cpu'):
        out = WheelRPMTorch(device)
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
        return "WheelRPMTorch from {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.data.cpu().numpy().round(4), self.stamp, self.device)
