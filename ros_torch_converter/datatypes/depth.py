"""
DepthImageTorch: a metric depth-image datatype for ros_torch_converter.

Why this exists: `main` has no depth converter, and `Float32bitImageTorch`
(on the parvm/tartan_rgbt branch) requires cv_bridge + raw 32FC1 Image. Our MultiSense
`openni_depth` stream is a PNG-compressed mono16 (uint16, millimetres) CompressedImage.
This class decodes it purely with cv2 (no cv_bridge, matching CompressedImageTorch),
preserves the raw integer depth (no 8-bit scaling), and records the metric scale so
downstream code can recover metres. It also accepts raw sensor_msgs/Image (16UC1/mono16
or 32FC1) for generality.

On disk (KITTI layout) depth is stored losslessly:
  - uint16 sources -> 16-bit PNG `{idx:08d}.png` (raw sensor units, e.g. mm) + `depth_scale` in info.yaml
  - float32 sources -> `{idx:08d}.npy` (metres) + `depth_scale: 1.0`
`from_kitti` reads either back as a single-channel float tensor of the *raw* units;
call `to_meters()` (or read `depth_scale`) to convert.
"""
import os

import cv2
import torch
import numpy as np

from sensor_msgs.msg import Image

from tartandriver_utils.ros_utils import stamp_to_time

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import (
    update_info_file,
    update_timestamp_file,
    read_info_file,
    read_timestamp_file,
)

# OpenNI/MultiSense depth is published in millimetres -> metres.
DEFAULT_DEPTH_SCALE = 1e-3


class DepthImageTorch(TorchCoordinatorDataType):
    """
    Metric depth image. Stores a single-channel HxWx1 float tensor of raw depth units
    (e.g. millimetres for openni). `depth_scale` converts a stored unit to metres.
    Invalid/no-return pixels are 0.
    """
    to_rosmsg_type = Image
    from_rosmsg_type = Image
    time_spec = TimeSpec.SYNC

    def __init__(self, device):
        super().__init__()
        self.image = torch.zeros(0, 0, 1, device=device)
        self.depth_scale = DEFAULT_DEPTH_SCALE
        self.device = device

    # ---- constructors --------------------------------------------------
    def from_numpy(image, device, depth_scale=DEFAULT_DEPTH_SCALE):
        res = DepthImageTorch(device=device)
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        res.image = torch.tensor(arr, dtype=torch.float32, device=device)
        res.depth_scale = depth_scale
        return res

    @staticmethod
    def _decode_msg(msg):
        """Return a single-channel numpy depth array + inferred depth_scale.

        Handles CompressedImage (png-compressed mono16) and raw Image (16UC1/mono16/32FC1).
        """
        # CompressedImage has a `format` field; raw Image has `encoding`.
        if hasattr(msg, "format"):
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Failed to decode compressed depth image")
            scale = DEFAULT_DEPTH_SCALE if img.dtype == np.uint16 else 1.0
            return img, scale

        # raw Image
        enc = getattr(msg, "encoding", "").lower()
        h, w = msg.height, msg.width
        if enc in ("32fc1",):
            img = np.frombuffer(msg.data, np.float32).reshape(h, w)
            scale = 1.0  # already metres
        elif enc in ("16uc1", "mono16"):
            img = np.frombuffer(msg.data, np.uint16).reshape(h, w)
            scale = DEFAULT_DEPTH_SCALE
        else:
            raise ValueError(f"Unsupported depth encoding: {msg.encoding!r}")
        return img, scale

    def from_rosmsg(msg, device="cpu", camera_info_torch=None, rectify=False):
        """Decode a depth message. `rectify`/`camera_info_torch` are accepted for API
        compatibility but ignored — openni depth is already in the rectified left frame,
        and depth must not be bilinearly remapped.
        """
        res = DepthImageTorch(device)
        img, scale = DepthImageTorch._decode_msg(msg)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        res.image = torch.from_numpy(img.astype(np.float32)).to(device)
        res.depth_scale = scale
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_meters(self):
        """Return the depth tensor in metres (raw units * depth_scale)."""
        return self.image * self.depth_scale

    # ---- (de)serialization --------------------------------------------
    def to_rosmsg(self, encoding="16UC1"):
        raise NotImplementedError("DepthImageTorch.to_rosmsg not needed for offline use")

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_info_file(base_dir, "frame_id", self.frame_id)
        update_info_file(base_dir, "depth_scale", float(self.depth_scale))

        img_np = self.image.detach().cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)

        # Integer sensor units (scale != 1.0, e.g. mm) -> lossless 16-bit PNG;
        # already-metric float depth (scale == 1.0) -> .npy.
        if self.depth_scale != 1.0:
            save_fp = os.path.join(base_dir, "{:08d}.png".format(idx))
            cv2.imwrite(save_fp, img_np.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            save_fp = os.path.join(base_dir, "{:08d}.npy".format(idx))
            np.save(save_fp, img_np.astype(np.float32))

    def from_kitti(base_dir, idx, device="cpu"):
        img = DepthImageTorch(device=device)
        png_fp = os.path.join(base_dir, "{:08d}.png".format(idx))
        npy_fp = os.path.join(base_dir, "{:08d}.npy".format(idx))
        if os.path.exists(png_fp):
            img_np = cv2.imread(png_fp, cv2.IMREAD_UNCHANGED)
        else:
            img_np = np.load(npy_fp)
        if img_np.ndim == 2:
            img_np = img_np[..., np.newaxis]
        img.image = torch.tensor(img_np.astype(np.float32), device=device)
        img.depth_scale = float(read_info_file(base_dir, "depth_scale", default_value=DEFAULT_DEPTH_SCALE))
        img.stamp = read_timestamp_file(base_dir, idx)
        img.frame_id = read_info_file(base_dir, "frame_id")
        return img

    # ---- misc ----------------------------------------------------------
    def to(self, device):
        self.device = device
        self.image = self.image.to(device)
        return self

    def rand_init(device="cpu"):
        data = torch.randint(65536, size=(480, 640, 1), device=device, dtype=torch.float32)
        out = DepthImageTorch.from_numpy(data.cpu().numpy(), device=device)
        out.frame_id = "random"
        out.stamp = float(np.random.rand())
        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if not np.isclose(self.depth_scale, other.depth_scale):
            return False
        if not torch.allclose(self.image, other.image):
            return False
        return True

    def __repr__(self):
        return "DepthImageTorch of shape {} (scale={}, time={:.2f}, frame={}, device={})".format(
            tuple(self.image.shape), self.depth_scale, self.stamp, self.frame_id, self.device
        )
