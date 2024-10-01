import torch

from std_msgs.msg import Float32

from ros_torch_converter.conversions.base import Conversion


class Float32ToFloatTensor(Conversion):
    """Convert a Float32"""

    def __init__(self):
        pass

    @property
    def msg_type(self):
        return Float32

    def cvt(self, msg):
        return torch.tensor([msg.data])
