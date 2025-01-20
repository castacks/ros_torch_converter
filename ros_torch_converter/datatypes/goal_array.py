import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

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
        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.goals.cpu().numpy())

    def from_kitti(self, base_dir, idx, device='cpu'):
        pass
    
    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def __repr__(self):
        return "GoalArrayTorch with shape {}, device {}".format(self.goals.shape, self.device)