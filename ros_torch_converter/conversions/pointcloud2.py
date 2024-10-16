import torch
import numpy as np
import ros2_numpy

from sensor_msgs.msg import PointCloud2

from ros_torch_converter.conversions.base import Conversion


class PointCloud2ToTorchPointCloud(Conversion):
    """Convert PointCloud2 to XYZ poitncloud

    note that this doesnt do arbitrary features yet
    """

    def __init__(self):
        pass

    @property
    def msg_type(self):
        return PointCloud2

    def cvt(self, msg):
        pcl_np = ros2_numpy.numpify(msg)
        xyz = np.stack(
            [pcl_np["x"].flatten(), pcl_np["y"].flatten(), pcl_np["z"].flatten()], axis=-1
        )

        return torch.from_numpy(xyz).float()