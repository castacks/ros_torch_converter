import torch

from geometry_msgs.msg import PoseArray

from ros_torch_converter.conversions.base import Conversion

class PoseArrayTo2DGoalArray(Conversion):
    """Convert a PoseArray to a 2d GoalArray
    """
    def __init__(self):
        pass
    
    @property
    def msg_type(self):
        return PoseArray
    
    def cvt(self, msg):
        res = []
        for pose in msg.poses:
            res.append(torch.tensor([
                pose.position.x,
                pose.position.y
            ]))
        return torch.stack(res, axis=0)