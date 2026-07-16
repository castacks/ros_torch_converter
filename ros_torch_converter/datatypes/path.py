import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

POSE_DIM = 7

class PathTorch(TorchCoordinatorDataType):
    """
    Coordinator type for Paths.

    Note that at the moment, we aren't using the time field for anything, but this is ok for now:
    https://answers.ros.org/question/299716/
    """
    to_rosmsg_type = Path
    from_rosmsg_type = Path
    time_spec = TimeSpec.SYNC

    def __init__(self, device='cpu'):
        super().__init__()
        self.poses = torch.zeros(0, POSE_DIM, device=device)
        self.device = device

    @staticmethod
    def from_torch(poses):
        if poses.ndim != 2 or poses.shape[1] != POSE_DIM:
            raise ValueError(
                "PathTorch.from_torch expected [P, {}], got {}".format(
                    POSE_DIM, tuple(poses.shape)
                )
            )
        pat = PathTorch(device=poses.device)
        pat.poses = poses
        return pat

    @staticmethod
    def from_rosmsg(msg, device='cpu'):
        pat = PathTorch(device=device)
        poses = np.empty((len(msg.poses), POSE_DIM), dtype=np.float32)
        for i, pose in enumerate(msg.poses):
            poses[i] = [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ]

        pat.poses = torch.from_numpy(poses).to(device)
        pat.stamp = stamp_to_time(msg.header.stamp)
        pat.frame_id = msg.header.frame_id

        return pat
    
    def to_rosmsg(self):
        msg = Path()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        for pose in self.poses:
            path_pose = PoseStamped()
            path_pose.header.stamp = msg.header.stamp
            path_pose.header.frame_id = msg.header.frame_id

            path_pose.pose.position.x = pose[0].item()
            path_pose.pose.position.y = pose[1].item()
            path_pose.pose.position.z = pose[2].item()
            path_pose.pose.orientation.x = pose[3].item()
            path_pose.pose.orientation.y = pose[4].item()
            path_pose.pose.orientation.z = pose[5].item()
            path_pose.pose.orientation.w = pose[6].item()

            msg.poses.append(path_pose)

        return msg
    
    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_info_file(base_dir, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.poses.cpu().numpy())

    @staticmethod
    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp)
        if data.size == 0:
            data = np.zeros((0, POSE_DIM), dtype=np.float32)
        elif data.ndim == 1:
            data = data.reshape(1, -1)
        data = torch.tensor(data, dtype=torch.float, device=device)

        gat = PathTorch.from_torch(data)
        
        gat.stamp = read_timestamp_file(base_dir, idx)
        gat.frame_id = read_info_file(base_dir,  'frame_id')

        return gat
    
    @staticmethod
    def rand_init(device='cpu'):
        goals = torch.rand(10, 7, device=device)
        gat = PathTorch.from_torch(goals)

        gat.frame_id = 'random'
        gat.stamp = np.random.rand()

        return gat

    def __eq__(self, other):
        if not isinstance(other, PathTorch):
            return NotImplemented

        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if self.poses.shape != other.poses.shape:
            return False

        if not torch.allclose(self.poses, other.poses):
            return False

        return True

    def to(self, device):
        self.device = device
        self.poses = self.poses.to(device)
        return self

    def __repr__(self):
        return "PathTorch with shape {}, device {}".format(self.poses.shape, self.device)
