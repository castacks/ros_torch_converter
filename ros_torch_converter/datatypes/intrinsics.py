import os
import yaml
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import update_frame_file, update_timestamp_file, read_frame_file, read_timestamp_file

from sensor_msgs.msg import CameraInfo

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class IntrinsicsTorch(TorchCoordinatorDataType):
    """
    Class for camera info. This consists of a 3x3 intrinsics matrix
    """
    to_rosmsg_type = CameraInfo
    from_rosmsg_type = CameraInfo

    def __init__(self, device='cpu'):
        super().__init__()
        self.intrinsics = torch.zeros(3, 3, device=device)
        self.device = device
    
    def from_rosmsg(msg, use_p=True, device='cpu'):
        res = IntrinsicsTorch(device=device)
        if use_p:
            res.intrinsics = torch.tensor(msg.p, device=device).reshape(3, 4)[:3, :3]
        else:
            res.intrinsics = torch.tensor(msg.k, device=device).reshape(3, 3)
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_torch(intrinsics):
        res = IntrinsicsTorch(device=intrinsics.device)
        res.intrinsics = intrinsics.float()
        return res

    def from_numpy(intrinsics, device):
        res = IntrinsicsTorch(device=device)
        res.intrinsics = torch.from_numpy(intrinsics, device=device, dtype=torch.float32)
        return res

    def to_rosmsg(self):
        msg = CameraInfo()

        msg.k = self.intrinsics.cpu().numpy().flatten().tolist()
        msg.r = np.eye(3).flatten().tolist()
        P = np.zeros((3, 4))
        P[:3, :3] = self.intrinsics.cpu().numpy()
        msg.p = P.flatten().tolist()

        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        return msg

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.intrinsics.cpu().numpy().flatten())

    def from_kitti(base_dir, idx, device='cpu'):
        res = IntrinsicsTorch(device=device)

        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp).reshape(3, 3)

        res.intrinsics = torch.tensor(data).float().to(device)
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return res

    def to(self, device):
        self.device = device
        self.intrinsics = self.intrinsics.to(device)
        return self
    
    def rand_init(device='cpu'):
        data = torch.eye(3, device=device)
        data[[0, 1, 0, 1], [0, 1, 2, 2]] = torch.rand(size=(4, )) * 100.

        out = IntrinsicsTorch.from_torch(data)
        out.frame_id = 'random'
        out.stamp = np.random.rand()

        return out

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.intrinsics, other.intrinsics):
            return False

        return True

    def __repr__(self):
        return "IntrinsicsTorch with k:\n{}, device = {}".format(self.intrinsics.cpu().numpy().round(4), self.device)


class CameraInfoTorch(TorchCoordinatorDataType):
    """
    Class for full camera info message including intrinsics, distortion, and rectification parameters.
    This is more complete than IntrinsicsTorch and includes all calibration data.
    """
    to_rosmsg_type = CameraInfo
    from_rosmsg_type = CameraInfo

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Camera intrinsic matrix K (3x3)
        self.k = torch.zeros(3, 3, device=device)
        # Distortion coefficients D (variable length)
        self.d = torch.zeros(0, device=device)
        # Rectification matrix R (3x3)
        self.r = torch.eye(3, device=device)
        # Projection matrix P (3x4)
        self.p = torch.zeros(3, 4, device=device)
        # Distortion model name
        self.distortion_model = ""
        # Image dimensions
        self.width = 0
        self.height = 0
    
    def from_rosmsg(msg, device='cpu'):
        res = CameraInfoTorch(device=device)
        res.k = torch.tensor(msg.k, device=device, dtype=torch.float32).reshape(3, 3)
        res.d = torch.tensor(msg.d, device=device, dtype=torch.float32)
        res.r = torch.tensor(msg.r, device=device, dtype=torch.float32).reshape(3, 3)
        res.p = torch.tensor(msg.p, device=device, dtype=torch.float32).reshape(3, 4)
        res.distortion_model = msg.distortion_model
        res.width = msg.width
        res.height = msg.height
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_rosmsg(self):
        msg = CameraInfo()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        
        msg.k = self.k.cpu().numpy().flatten().tolist()
        msg.d = self.d.cpu().numpy().tolist()
        msg.r = self.r.cpu().numpy().flatten().tolist()
        msg.p = self.p.cpu().numpy().flatten().tolist()
        msg.distortion_model = self.distortion_model
        msg.width = self.width
        msg.height = self.height
        
        return msg

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        # Save as YAML for better human readability and to preserve all info
        save_fp = os.path.join(base_dir, "{:08d}.yaml".format(idx))
        data = {
            'k': self.k.cpu().numpy().tolist(),
            'd': self.d.cpu().numpy().tolist(),
            'r': self.r.cpu().numpy().tolist(),
            'p': self.p.cpu().numpy().tolist(),
            'distortion_model': self.distortion_model,
            'width': self.width,
            'height': self.height,
        }
        with open(save_fp, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def from_kitti(base_dir, idx, device='cpu'):
        res = CameraInfoTorch(device=device)

        fp = os.path.join(base_dir, "{:08d}.yaml".format(idx))
        with open(fp, 'r') as f:
            data = yaml.safe_load(f)

        res.k = torch.tensor(data['k'], device=device, dtype=torch.float32)
        res.d = torch.tensor(data['d'], device=device, dtype=torch.float32)
        res.r = torch.tensor(data['r'], device=device, dtype=torch.float32)
        res.p = torch.tensor(data['p'], device=device, dtype=torch.float32)
        res.distortion_model = data['distortion_model']
        res.width = data['width']
        res.height = data['height']
        
        res.stamp = read_timestamp_file(base_dir, idx)
        res.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return res

    def to(self, device):
        self.device = device
        self.k = self.k.to(device)
        self.d = self.d.to(device)
        self.r = self.r.to(device)
        self.p = self.p.to(device)
        return self
    
    def rand_init(device='cpu'):
        res = CameraInfoTorch(device=device)
        # Random intrinsics
        res.k = torch.eye(3, device=device)
        res.k[[0, 1, 0, 1], [0, 1, 2, 2]] = torch.rand(size=(4,), device=device) * 100.
        # Random distortion (5 coefficients for plumb_bob)
        res.d = torch.randn(5, device=device) * 0.1
        # Identity rectification
        res.r = torch.eye(3, device=device)
        # Random projection
        res.p = torch.zeros(3, 4, device=device)
        res.p[:3, :3] = res.k
        res.distortion_model = 'plumb_bob'
        res.width = 640
        res.height = 480
        res.frame_id = 'random'
        res.stamp = np.random.rand()
        return res

    def __eq__(self, other):
        if not isinstance(other, CameraInfoTorch):
            return False
        if self.frame_id != other.frame_id:
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if not torch.allclose(self.k, other.k):
            return False
        if not torch.allclose(self.d, other.d):
            return False
        if not torch.allclose(self.r, other.r):
            return False
        if not torch.allclose(self.p, other.p):
            return False
        if self.distortion_model != other.distortion_model:
            return False
        if self.width != other.width or self.height != other.height:
            return False
        return True

    def __repr__(self):
        return (f"CameraInfoTorch({self.width}x{self.height}, "
                f"model={self.distortion_model}, device={self.device})")