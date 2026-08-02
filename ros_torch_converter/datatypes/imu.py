import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import (
    update_frame_file, update_timestamp_file,
    read_frame_file, read_timestamp_file
)

from sensor_msgs.msg import Imu


class ImuTorch(TorchCoordinatorDataType):
    """
    TorchCoordinator wrapper for sensor_msgs/Imu.

    Stores angular velocity (gyro), linear acceleration (accel) and orientation.
    Intended to be extracted at full (native) rate via the `full_rate_topics`
    path in ros2bag_2_kitti.py (see ImuTorch.write_full_rate), NOT synced to the
    camera master clock.

    KITTI layout written by write_full_rate():
        <name>/timestamps.txt   # [N]      message header stamps (sec)
        <name>/data.txt         # [N, 6]   wx wy wz ax ay az
        <name>/orientation.txt  # [N, 4]   qx qy qz qw (identity when unavailable, e.g. data_raw)
        <name>/frames.yaml      # frame_id
    """
    to_rosmsg_type = Imu
    from_rosmsg_type = Imu

    # column order of data.txt
    DATA_COLUMNS = ['wx', 'wy', 'wz', 'ax', 'ay', 'az']

    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        self.gyro = torch.zeros(3, device=device)          # angular_velocity
        self.accel = torch.zeros(3, device=device)         # linear_acceleration
        self.orientation = torch.tensor([0., 0., 0., 1.], device=device)  # qx qy qz qw
        self.device = device

    @staticmethod
    def from_rosmsg(msg, device='cpu'):
        from tartandriver_utils.ros_utils import stamp_to_time

        res = ImuTorch(device=device)
        res.gyro = torch.tensor([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], device=device)
        res.accel = torch.tensor([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], device=device)
        res.orientation = torch.tensor([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ], device=device)

        if hasattr(msg, 'header'):
            res.stamp = stamp_to_time(msg.header.stamp)
            res.frame_id = msg.header.frame_id

        return res

    def to_rosmsg(self):
        msg = Imu()
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = \
            [x.item() for x in self.gyro]
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = \
            [x.item() for x in self.accel]
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = \
            [x.item() for x in self.orientation]
        return msg

    def to(self, device):
        self.device = device
        self.gyro = self.gyro.to(device)
        self.accel = self.accel.to(device)
        self.orientation = self.orientation.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        Per-index write (kept for interface parity). Grows data.txt/orientation.txt.
        NOTE: O(n^2) over a full run - use write_full_rate() for native-rate IMU.
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        self._write_row(os.path.join(base_dir, "data.txt"), idx,
                        np.concatenate([self.gyro.cpu().numpy(), self.accel.cpu().numpy()]))
        self._write_row(os.path.join(base_dir, "orientation.txt"), idx,
                        self.orientation.cpu().numpy())

    @staticmethod
    def _write_row(save_fp, idx, row):
        ncol = row.shape[0]
        if not os.path.exists(save_fp):
            data = np.full([idx + 1, ncol], fill_value=np.inf)
        else:
            data = np.loadtxt(save_fp).reshape(-1, ncol)

        if data.shape[0] < (idx + 1):
            data_new = np.full([idx + 1, ncol], fill_value=np.inf)
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = row
        np.savetxt(save_fp, data)

    @staticmethod
    def write_full_rate(base_dir, samples):
        """
        Efficient single-shot write for a full list of ImuTorch samples (sorted by stamp).

        Args:
            base_dir: output dir for this modality (e.g. <dst>/zed_imu_raw)
            samples:  list[ImuTorch]
        """
        os.makedirs(base_dir, exist_ok=True)

        if len(samples) == 0:
            print('warning: no IMU samples to write for {}'.format(base_dir))
            return

        order = np.argsort([s.stamp for s in samples])
        samples = [samples[i] for i in order]

        stamps = np.array([s.stamp for s in samples])
        data = np.stack([
            np.concatenate([s.gyro.cpu().numpy(), s.accel.cpu().numpy()]) for s in samples
        ], axis=0)
        orient = np.stack([s.orientation.cpu().numpy() for s in samples], axis=0)

        np.savetxt(os.path.join(base_dir, 'timestamps.txt'), stamps)
        np.savetxt(os.path.join(base_dir, 'data.txt'), data)
        np.savetxt(os.path.join(base_dir, 'orientation.txt'), orient)
        update_frame_file(base_dir, 0, 'frame_id', samples[0].frame_id)

    @staticmethod
    def from_kitti(base_dir, idx, device='cpu'):
        data = np.loadtxt(os.path.join(base_dir, "data.txt")).reshape(-1, 6)[idx]
        orient = np.loadtxt(os.path.join(base_dir, "orientation.txt")).reshape(-1, 4)[idx]

        out = ImuTorch(device=device)
        out.gyro = torch.tensor(data[:3], device=device).float()
        out.accel = torch.tensor(data[3:], device=device).float()
        out.orientation = torch.tensor(orient, device=device).float()

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_frame_file(base_dir, idx, 'frame_id')
        return out

    @staticmethod
    def rand_init(device='cpu'):
        out = ImuTorch(device=device)
        out.gyro = torch.rand(3, device=device)
        out.accel = torch.rand(3, device=device)
        out.orientation = torch.tensor([0., 0., 0., 1.], device=device)
        out.frame_id = 'random'
        out.stamp = np.random.rand()
        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if not torch.allclose(self.gyro, other.gyro):
            return False
        if not torch.allclose(self.accel, other.accel):
            return False
        if not torch.allclose(self.orientation, other.orientation):
            return False
        return True

    def __repr__(self):
        return "ImuTorch gyro={}, accel={}, device {}".format(
            self.gyro.cpu().numpy(), self.accel.cpu().numpy(), self.device)
