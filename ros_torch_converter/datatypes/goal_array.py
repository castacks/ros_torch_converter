import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import update_frame_file, update_timestamp_file, read_frame_file, read_timestamp_file

from geometry_msgs.msg import PoseArray, Pose, Point

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class GoalArrayTorch(TorchCoordinatorDataType):
    """
    Set of xyz coordinate goals
    """
    to_rosmsg_type = PoseArray
    from_rosmsg_type = PoseArray

    def __init__(self, device='cpu'):
        super().__init__()
        self.goals = torch.zeros(0, 3, device=device)
        self.device = device
    
    def from_torch(goals):
        gat = GoalArrayTorch(device=goals.device)
        gat.goals = goals
        return gat

    def from_rosmsg(msg, device='cpu'):
        res = GoalArrayTorch(device)
        
        for pose in msg.poses:
            new_goal = torch.tensor([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ], 
            device=res.device).float().reshape(1, 3)

            res.goals = torch.cat([res.goals, new_goal], dim=0)

        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res
    
    def to_rosmsg(self):
        poses = []
        for goal in self.goals:
            pt = Point(
                x=goal[0].item(),
                y=goal[1].item(),
                z=goal[2].item(),
            )
            pose = Pose(position=pt)
            poses.append(pose)

        msg = PoseArray(poses=poses)
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        return msg

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.goals.cpu().numpy())

    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp)
        data = torch.tensor(data, dtype=torch.float, device=device)

        gat = GoalArrayTorch.from_torch(data)
        
        gat.stamp = read_timestamp_file(base_dir, idx)
        gat.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return gat
    
    def rand_init(device='cpu'):
        goals = torch.rand(10, 3, device=device)
        gat = GoalArrayTorch.from_torch(goals)

        gat.frame_id = 'random'
        gat.stamp = np.random.rand()

        return gat

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.goals, other.goals):
            return False

        return True

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def __repr__(self):
        return "GoalArrayTorch with shape {}, device {}".format(self.goals.shape, self.device)