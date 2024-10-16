import torch

from sensor_msgs.msg import CameraInfo

from ros_torch_converter.conversions.base import Conversion


class CameraInfoToIntrinsics(Conversion):
    """Convert Odometry to 7d pose

    Stacked as [x,y,z, qx,qy,qz,qw]
    """

    def __init__(self, use_p=True):
        self.use_p = use_p

    @property
    def msg_type(self):
        return CameraInfo

    def cvt(self, msg):
        if self.use_p:
            return torch.tensor(msg.p).reshape(3, 4)[:3, :3]
        else:

            return torch.tensor(msg.k).reshape(3, 3)