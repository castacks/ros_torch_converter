import os
import torch
import numpy as np

from std_msgs.msg import Bool
from core_interfaces.msg import BoolStamped

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import (
    update_info_file,
    update_timestamp_file,
    read_info_file,
    read_timestamp_file,
)


class BoolTorch(TorchCoordinatorDataType):
    """Wrapper for boolean ROS messages.

    Castacks main uses core_interfaces/BoolStamped on the wire.  The
    deserializer also accepts std_msgs/Bool so the strapsai 3D configs can be
    decoded when messages arrive from bag/conversion utilities.
    """

    to_rosmsg_type = BoolStamped
    from_rosmsg_type = BoolStamped
    time_spec = TimeSpec.SYNC  # interpolating bools is nonsense

    def __init__(self, device="cpu"):
        super().__init__()
        self.child_frame_id = ""
        self.data = torch.zeros(1, dtype=torch.bool, device=device)
        self.device = device

    def from_torch(x):
        out = BoolTorch(device=x.device)
        out.data = x.to(dtype=torch.bool).reshape(1)
        return out

    def from_rosmsg(msg, device="cpu"):
        res = BoolTorch(device=device)
        res.data = torch.tensor([bool(msg.data)], dtype=torch.bool, device=device)
        if hasattr(msg, "header"):
            res.stamp = stamp_to_time(msg.header.stamp)
            res.frame_id = msg.header.frame_id
        return res

    def to_rosmsg(self):
        msg = BoolStamped()
        msg.data = bool(self.data.item())
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes should be stored as rows of a matrix
        """
        update_timestamp_file(base_dir, idx, self.stamp, file="timestamps.txt")
        update_info_file(base_dir, "frame_id", self.frame_id)
        self.save_to_file(base_dir, idx, file="data.txt")

    def save_to_file(self, base_dir, idx, file="data.txt"):
        save_fp = os.path.join(base_dir, file)
        if not os.path.exists(save_fp):
            data = np.zeros([idx + 1], dtype=bool)
        else:
            data = np.loadtxt(save_fp).reshape(-1).astype(bool)

        if data.shape[0] < (idx + 1):
            data_new = np.zeros([idx + 1], dtype=bool)
            data_new[: data.shape[0]] = data
            data = data_new

        data[idx] = self.data.cpu().numpy().item()
        np.savetxt(save_fp, data, fmt="%d")

    def from_kitti(base_dir, idx, device="cpu"):
        fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(fp).reshape(-1)[idx]

        out = BoolTorch(device=device)
        out.data = torch.tensor([bool(data)], device=device)

        out.stamp = read_timestamp_file(base_dir, idx)
        out.frame_id = read_info_file(base_dir, "frame_id")

        return out

    def from_kitti_multi(base_dir, idxs, device="cpu"):
        fp = os.path.join(base_dir, "data.txt")

        data = np.loadtxt(fp).reshape(-1)[idxs]
        data = torch.tensor(data, device=device).bool()
        out = []
        for x in data:
            bt = BoolTorch(device=device)
            bt.data = x.reshape(1)
            out.append(bt)
        return out

    def rand_init(device="cpu"):
        out = BoolTorch(device=device)
        out.data = torch.randint(0, 2, size=(1,), device=device).bool()
        out.frame_id = "random"
        out.stamp = np.random.rand()

        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.equal(self.data, other.data):
            return False

        return True

    def __repr__(self):
        return "BoolTorch with data {}, device {}".format(
            self.data.item(),
            self.device,
        )
