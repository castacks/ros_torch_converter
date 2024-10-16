import torch
import cv_bridge

from sensor_msgs.msg import Image

from ros_torch_converter.conversions.base import Conversion


class ImageToTorchImage(Conversion):
    """Convert Odometry to 7d pose

    Stacked as [x,y,z, qx,qy,qz,qw]
    """

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

    @property
    def msg_type(self):
        return Image

    def cvt(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg)
        return torch.from_numpy(img/255.).float()