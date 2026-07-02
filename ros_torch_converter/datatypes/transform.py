import os
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from tartandriver_utils.geometry_utils import pose_to_htm, htm_to_pose
from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class TransformTorch(TorchCoordinatorDataType):
    """
    Datatype for pose. Represent as a 4x4 HTM
    """
    to_rosmsg_type = TransformStamped
    from_rosmsg_type = TransformStamped
    time_spec = TimeSpec.INTERP

    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        self.transform = torch.zeros(4, 4, device=device)
        self.device = device
    
    def from_rosmsg(msg, device):
        res = TransformTorch(device=device)
        pq = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w,
        ])
        H = torch.from_numpy(pose_to_htm(pq)).float().to(device)
        res.transform = H
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        res.child_frame_id = msg.child_frame_id
        return res

    def to_rosmsg(self):
        msg = TransformStamped()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id
        pose = htm_to_pose(self.transform.numpy())
        msg.transform.translation.x = pose[0].item()
        msg.transform.translation.y = pose[1].item()
        msg.transform.translation.z = pose[2].item()

        msg.transform.rotation.x = pose[3].item()
        msg.transform.rotation.y = pose[4].item()
        msg.transform.rotation.z = pose[5].item()
        msg.transform.rotation.w = pose[6].item()

        return msg
    
    def from_numpy(transform, child_frame_id, device='cpu'):
        res = TransformTorch(device=device)
        res.transform = torch.from_numpy(transform, dtype=torch.float32).to(device)
        res.child_frame_id = child_frame_id
        return res
    
    def from_torch(transform, child_frame_id):
        res = TransformTorch(device=transform.device)
        res.transform = transform
        res.child_frame_id = child_frame_id
        return res

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp, file='timestamps.txt')
        update_info_file(base_dir, 'frame_id', self.frame_id)
        update_info_file(base_dir, 'child_frame_id', self.child_frame_id)
        self.save_to_file(base_dir, idx, file='data.txt')

    def to_kitti_interp(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp, file='interp_timestamps.txt')
        update_info_file(base_dir, 'frame_id', self.frame_id)
        update_info_file(base_dir, 'child_frame_id', self.child_frame_id)
        self.save_to_file(base_dir, idx, file='interp_data.txt')

    def save_to_file(self, base_dir, idx, file='data.txt'):
        _data = self.transform[:3].flatten().cpu().numpy()

        save_fp = os.path.join(base_dir, file)
        if not os.path.exists(save_fp):
            data = float('inf') * np.ones([idx+1, 12])
        else:
            #need to reshape for 1-row data
            data = np.loadtxt(save_fp).reshape(-1, 12)

        if data.shape[0] < (idx+1):
            data_new = float('inf') * np.ones([idx+1, 12])
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = _data

        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        """define how to convert this dtype from a kitti file
        """
        fp = os.path.join(base_dir, "data.txt")
        _data = np.loadtxt(fp).reshape(-1, 12)[idx]
        H = np.eye(4)
        H[:3] = _data.reshape(3, 4)
        H = torch.tensor(H, dtype=torch.float, device=device)

        child_frame_id = read_info_file(base_dir, 'child_frame_id')

        tft = TransformTorch.from_torch(H, child_frame_id)

        tft.stamp = read_timestamp_file(base_dir, idx)
        tft.frame_id = read_info_file(base_dir,  'frame_id')

        return tft
        
    def rand_init(device='cpu'):
        angs = np.random.rand(3) * 2*np.pi
        rotm = R.from_euler('xyz', angs).as_matrix()
        rotm = torch.tensor(rotm, dtype=torch.float, device=device)

        trans = torch.rand(3)

        H = torch.eye(4)
        H[:3, :3] = rotm
        H[:3, -1] = trans

        tft = TransformTorch.from_torch(H, 'random_child')
        tft.frame_id = 'random'
        tft.stamp = np.random.rand()

        return tft

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if self.child_frame_id != other.child_frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.transform, other.transform):
            return False

        return True

    def to(self, device):
        self.device = device
        self.transform = self.transform.to(device)
        return self

    def __repr__(self):
        return "TransformTorch from {} to {} with H:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.child_frame_id, self.transform.cpu().numpy().round(4), self.stamp, self.device)
    
class OdomTransformTorch(TransformTorch):
    """Same as transformtorch, but to/from nav_msgs/Odometry
    """
    to_rosmsg_type = Odometry
    from_rosmsg_type = Odometry
    time_spec = TimeSpec.INTERP
    
    def from_rosmsg(msg, device):
        res = TransformTorch(device=device)
        pq = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])
        H = torch.from_numpy(pose_to_htm(pq)).float().to(device)
        res.transform = H
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        res.child_frame_id = msg.child_frame_id
        return res

    def to_rosmsg(self):
        msg = Odometry()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id
        pose = htm_to_pose(self.transform.numpy())
        msg.pose.pose.position.x = pose[0].item()
        msg.pose.pose.position.y = pose[1].item()
        msg.pose.pose.position.z = pose[2].item()

        msg.pose.pose.orientation.x = pose[3].item()
        msg.pose.pose.orientation.y = pose[4].item()
        msg.pose.pose.orientation.z = pose[5].item()
        msg.pose.pose.orientation.w = pose[6].item()

        return msg