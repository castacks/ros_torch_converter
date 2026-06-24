from __future__ import annotations

import numpy as np
import torch

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from tartandriver_utils.ros_utils import time_to_stamp


def _pack_rgb_float32(rgb01: np.ndarray) -> np.ndarray:
    rgb255 = np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint32)
    packed = (rgb255[:, 0] << 16) | (rgb255[:, 1] << 8) | rgb255[:, 2]
    return packed.view(np.float32)


def _base_cloud(stamp, frame_id: str, width: int) -> PointCloud2:
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = time_to_stamp(stamp)
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(width)
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 20
    msg.row_step = msg.point_step * msg.width
    return msg


class SemanticClassVoxelCloudTorch(TorchCoordinatorDataType):
    """RViz-friendly semantic class voxel centers as PointCloud2.

    Payload layout is exactly 20 bytes/point:
      x@0 f32, y@4 f32, z@8 f32, rgb@12 f32, class_id@16 int8, pad@17..19.
    """

    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.xyz = torch.zeros(0, 3, dtype=torch.float32, device=device)
        self.rgb = torch.zeros(0, 3, dtype=torch.float32, device=device)
        self.class_id = torch.zeros(0, dtype=torch.int8, device=device)

    @staticmethod
    def from_torch(xyz, rgb, class_id):
        res = SemanticClassVoxelCloudTorch(device=xyz.device)
        res.xyz = xyz.to(dtype=torch.float32)
        res.rgb = rgb.to(dtype=torch.float32)
        res.class_id = class_id.to(dtype=torch.int8)
        return res

    def to_rosmsg(self):
        n = int(self.xyz.shape[0])
        msg = _base_cloud(self.stamp, self.frame_id, n)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=16, datatype=PointField.INT8, count=1),
        ]
        dtype = np.dtype(
            {
                "names": ["x", "y", "z", "rgb", "class_id"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "i1"],
                "offsets": [0, 4, 8, 12, 16],
                "itemsize": 20,
            }
        )
        arr = np.zeros(n, dtype=dtype)
        if n:
            xyz = self.xyz.detach().cpu().numpy().astype(np.float32)
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            arr["rgb"] = _pack_rgb_float32(self.rgb.detach().cpu().numpy())
            arr["class_id"] = self.class_id.detach().cpu().numpy().astype(np.int8)
        msg.data = arr.tobytes()
        return msg

    def to(self, device):
        self.device = device
        self.xyz = self.xyz.to(device)
        self.rgb = self.rgb.to(device)
        self.class_id = self.class_id.to(device)
        return self


class SemanticEntropyVoxelCloudTorch(TorchCoordinatorDataType):
    """RViz-friendly semantic entropy voxel centers as PointCloud2.

    Payload layout is exactly 20 bytes/point:
      x@0 f32, y@4 f32, z@8 f32, rgb@12 f32, entropy@16 f32.
    """

    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.xyz = torch.zeros(0, 3, dtype=torch.float32, device=device)
        self.rgb = torch.zeros(0, 3, dtype=torch.float32, device=device)
        self.entropy = torch.zeros(0, dtype=torch.float32, device=device)

    @staticmethod
    def from_torch(xyz, rgb, entropy):
        res = SemanticEntropyVoxelCloudTorch(device=xyz.device)
        res.xyz = xyz.to(dtype=torch.float32)
        res.rgb = rgb.to(dtype=torch.float32)
        res.entropy = entropy.to(dtype=torch.float32)
        return res

    def to_rosmsg(self):
        n = int(self.xyz.shape[0])
        msg = _base_cloud(self.stamp, self.frame_id, n)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="entropy", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        dtype = np.dtype(
            {
                "names": ["x", "y", "z", "rgb", "entropy"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<f4"],
                "offsets": [0, 4, 8, 12, 16],
                "itemsize": 20,
            }
        )
        arr = np.zeros(n, dtype=dtype)
        if n:
            xyz = self.xyz.detach().cpu().numpy().astype(np.float32)
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            arr["rgb"] = _pack_rgb_float32(self.rgb.detach().cpu().numpy())
            arr["entropy"] = self.entropy.detach().cpu().numpy().astype(np.float32)
        msg.data = arr.tobytes()
        return msg

    def to(self, device):
        self.device = device
        self.xyz = self.xyz.to(device)
        self.rgb = self.rgb.to(device)
        self.entropy = self.entropy.to(device)
        return self
