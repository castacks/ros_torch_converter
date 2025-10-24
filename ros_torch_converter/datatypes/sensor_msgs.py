import os
import numpy as np
import torch

from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import (
    update_frame_file,
    update_timestamp_file,
    read_frame_file,
    read_timestamp_file,
)


class ImuTorch(TorchCoordinatorDataType):
    """TorchCoordinator class for IMU messages"""

    from_rosmsg_type = Imu
    to_rosmsg_type = Imu

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.orientation = None  # quaternion [x, y, z, w]
        self.angular_velocity = None  # [x, y, z]
        self.linear_acceleration = None  # [x, y, z]
        self.stamp = 0.0
        self.frame_id = ""

    def from_rosmsg(msg, device="cpu"):
        res = ImuTorch(device)
        res.orientation = torch.tensor(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            device=device,
            dtype=torch.float32,
        )
        res.angular_velocity = torch.tensor(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            device=device,
            dtype=torch.float32,
        )
        res.linear_acceleration = torch.tensor(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            device=device,
            dtype=torch.float32,
        )
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, "frame_id", self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = torch.cat(
            [self.orientation, self.angular_velocity, self.linear_acceleration]
        )
        np.savetxt(save_fp, data.cpu().numpy())

    def from_kitti(base_dir, idx, device="cpu"):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = torch.tensor(np.loadtxt(fp), device=device, dtype=torch.float32)

        res = ImuTorch(device=device)
        res.orientation = data[:4]
        res.angular_velocity = data[4:7]
        res.linear_acceleration = data[7:10]
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, "frame_id")
        return res

    def to_rosmsg(self):
        msg = Imu()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        msg.orientation.x = float(self.orientation[0])
        msg.orientation.y = float(self.orientation[1])
        msg.orientation.z = float(self.orientation[2])
        msg.orientation.w = float(self.orientation[3])
        msg.angular_velocity.x = float(self.angular_velocity[0])
        msg.angular_velocity.y = float(self.angular_velocity[1])
        msg.angular_velocity.z = float(self.angular_velocity[2])
        msg.linear_acceleration.x = float(self.linear_acceleration[0])
        msg.linear_acceleration.y = float(self.linear_acceleration[1])
        msg.linear_acceleration.z = float(self.linear_acceleration[2])
        return msg

    def to(self, device):
        self.device = device
        self.orientation = self.orientation.to(device)
        self.angular_velocity = self.angular_velocity.to(device)
        self.linear_acceleration = self.linear_acceleration.to(device)
        return self

    def __eq__(self, other):
        if not isinstance(other, ImuTorch):
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if self.frame_id != other.frame_id:
            return False
        return (
            torch.allclose(self.orientation, other.orientation)
            and torch.allclose(self.angular_velocity, other.angular_velocity)
            and torch.allclose(self.linear_acceleration, other.linear_acceleration)
        )

    def rand_init(device="cpu"):
        res = ImuTorch(device=device)
        res.orientation = torch.randn(4, device=device)
        res.orientation = res.orientation / torch.norm(res.orientation)  # normalize
        res.angular_velocity = torch.randn(3, device=device)
        res.linear_acceleration = torch.randn(3, device=device)
        res.stamp = np.random.rand()
        res.frame_id = "random"
        return res


class NavSatFixTorch(TorchCoordinatorDataType):
    """TorchCoordinator class for NavSatFix (GPS) messages"""

    from_rosmsg_type = NavSatFix
    to_rosmsg_type = NavSatFix

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.stamp = 0.0
        self.frame_id = ""

    def from_rosmsg(msg, device="cpu"):
        res = NavSatFixTorch(device)
        res.latitude = msg.latitude
        res.longitude = msg.longitude
        res.altitude = msg.altitude
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, "frame_id", self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.array([self.latitude, self.longitude, self.altitude])
        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device="cpu"):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp)

        res = NavSatFixTorch(device=device)
        res.latitude = data[0]
        res.longitude = data[1]
        res.altitude = data[2]
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, "frame_id")
        return res

    def to_rosmsg(self):
        msg = NavSatFix()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        msg.latitude = self.latitude
        msg.longitude = self.longitude
        msg.altitude = self.altitude
        return msg

    def to(self, device):
        self.device = device
        return self

    def __eq__(self, other):
        if not isinstance(other, NavSatFixTorch):
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if self.frame_id != other.frame_id:
            return False
        return (
            abs(self.latitude - other.latitude) < 1e-8
            and abs(self.longitude - other.longitude) < 1e-8
            and abs(self.altitude - other.altitude) < 1e-8
        )

    def rand_init(device="cpu"):
        res = NavSatFixTorch(device=device)
        res.latitude = np.random.uniform(-90, 90)
        res.longitude = np.random.uniform(-180, 180)
        res.altitude = np.random.uniform(-100, 5000)
        res.stamp = np.random.rand()
        res.frame_id = "random"
        return res


class PoseWithCovarianceTorch(TorchCoordinatorDataType):
    """TorchCoordinator class for PoseWithCovarianceStamped messages"""

    from_rosmsg_type = PoseWithCovarianceStamped
    to_rosmsg_type = PoseWithCovarianceStamped

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.position = None  # [x, y, z]
        self.orientation = None  # quaternion [x, y, z, w]
        self.stamp = 0.0
        self.frame_id = ""

    def from_rosmsg(msg, device="cpu"):
        res = PoseWithCovarianceTorch(device)
        res.position = torch.tensor(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            device=device,
            dtype=torch.float32,
        )
        res.orientation = torch.tensor(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            device=device,
            dtype=torch.float32,
        )
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, "frame_id", self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = torch.cat([self.position, self.orientation])
        np.savetxt(save_fp, data.cpu().numpy())

    def from_kitti(base_dir, idx, device="cpu"):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = torch.tensor(np.loadtxt(fp), device=device, dtype=torch.float32)

        res = PoseWithCovarianceTorch(device=device)
        res.position = data[:3]
        res.orientation = data[3:7]
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, "frame_id")
        return res

    def to_rosmsg(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        msg.pose.pose.position.x = float(self.position[0])
        msg.pose.pose.position.y = float(self.position[1])
        msg.pose.pose.position.z = float(self.position[2])
        msg.pose.pose.orientation.x = float(self.orientation[0])
        msg.pose.pose.orientation.y = float(self.orientation[1])
        msg.pose.pose.orientation.z = float(self.orientation[2])
        msg.pose.pose.orientation.w = float(self.orientation[3])
        return msg

    def to(self, device):
        self.device = device
        self.position = self.position.to(device)
        self.orientation = self.orientation.to(device)
        return self

    def __eq__(self, other):
        if not isinstance(other, PoseWithCovarianceTorch):
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if self.frame_id != other.frame_id:
            return False
        return torch.allclose(self.position, other.position) and torch.allclose(
            self.orientation, other.orientation
        )

    def rand_init(device="cpu"):
        res = PoseWithCovarianceTorch(device=device)
        res.position = torch.randn(3, device=device)
        res.orientation = torch.randn(4, device=device)
        res.orientation = res.orientation / torch.norm(res.orientation)  # normalize
        res.stamp = np.random.rand()
        res.frame_id = "random"
        return res


class TwistTorch(TorchCoordinatorDataType):
    """TorchCoordinator class for TwistStamped messages"""

    from_rosmsg_type = TwistStamped
    to_rosmsg_type = TwistStamped

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.linear = None  # [x, y, z]
        self.angular = None  # [x, y, z]
        self.stamp = 0.0
        self.frame_id = ""

    def from_rosmsg(msg, device="cpu"):
        res = TwistTorch(device)
        res.linear = torch.tensor(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
            device=device,
            dtype=torch.float32,
        )
        res.angular = torch.tensor(
            [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z],
            device=device,
            dtype=torch.float32,
        )
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, "frame_id", self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = torch.cat([self.linear, self.angular])
        np.savetxt(save_fp, data.cpu().numpy())

    def from_kitti(base_dir, idx, device="cpu"):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = torch.tensor(np.loadtxt(fp), device=device, dtype=torch.float32)

        res = TwistTorch(device=device)
        res.linear = data[:3]
        res.angular = data[3:6]
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, "frame_id")
        return res

    def to_rosmsg(self):
        msg = TwistStamped()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = float(self.linear[0])
        msg.twist.linear.y = float(self.linear[1])
        msg.twist.linear.z = float(self.linear[2])
        msg.twist.angular.x = float(self.angular[0])
        msg.twist.angular.y = float(self.angular[1])
        msg.twist.angular.z = float(self.angular[2])
        return msg

    def to(self, device):
        self.device = device
        self.linear = self.linear.to(device)
        self.angular = self.angular.to(device)
        return self

    def __eq__(self, other):
        if not isinstance(other, TwistTorch):
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if self.frame_id != other.frame_id:
            return False
        return torch.allclose(self.linear, other.linear) and torch.allclose(
            self.angular, other.angular
        )

    def rand_init(device="cpu"):
        res = TwistTorch(device=device)
        res.linear = torch.randn(3, device=device)
        res.angular = torch.randn(3, device=device)
        res.stamp = np.random.rand()
        res.frame_id = "random"
        return res
