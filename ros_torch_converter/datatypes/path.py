import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

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
        self.poses = torch.zeros(0, 7, device=device)
        self.device = device

    def from_torch(poses):
        pat = PathTorch(device=poses.device)
        pat.poses = poses
        return pat
    
    def from_rosmsg(msg, device):
        pat = PathTorch(device=device)
        poses = []
        for _pose in msg.poses:
            poses.append(torch.tensor([
                _pose.pose.position.x,
                _pose.pose.position.y,
                _pose.pose.position.z,
                _pose.pose.orientation.x,
                _pose.pose.orientation.y,
                _pose.pose.orientation.z,
                _pose.pose.orientation.w,
            ]))
        poses = torch.stack(poses, dim=0)

        pat.poses = poses.to(device)
        pat.stamp = stamp_to_time(msg.header.stamp)
        pat.frame_id = msg.header.frame_id

        return pat
    
    def to_rosmsg(self):
        msg = Path()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        for _pose in self.poses:
            _path_pose = PoseStamped()
            _path_pose.stamp = msg.header.stamp
            _path_pose.frame_id = msg.header.frame_id

            _path_pose.pose.position.x = _pose[0].item()
            _path_pose.pose.position.y = _pose[1].item()
            _path_pose.pose.position.z = _pose[2].item()
            _path_pose.pose.orientation.x = _pose[3].item()
            _path_pose.pose.orientation.y = _pose[4].item()
            _path_pose.pose.orientation.z = _pose[5].item()
            _path_pose.pose.orientation.w = _pose[6].item()

            msg.poses.append(_path_pose)

        return msg
    
    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_info_file(base_dir, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.poses.cpu().numpy())

    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp)
        data = torch.tensor(data, dtype=torch.float, device=device)

        gat = PathTorch.from_torch(data)
        
        gat.stamp = read_timestamp_file(base_dir, idx)
        gat.frame_id = read_info_file(base_dir,  'frame_id')

        return gat
    
    def rand_init(device='cpu'):
        goals = torch.rand(10, 7, device=device)
        gat = PathTorch.from_torch(goals)

        gat.frame_id = 'random'
        gat.stamp = np.random.rand()

        return gat

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
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