from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from std_msgs.msg import Bool

class BoolTorch(TorchCoordinatorDataType):
    """
    Wrapper for std_msgs/Bool
    """
    to_rosmsg_type = Bool
    from_rosmsg_type = Bool

    def __init__(self, device='cpu'):
        super().__init__()
        self.data = False
        self.device = device

    @classmethod
    def from_rosmsg(cls, msg, device='cpu'):
        res = cls(device=device)
        res.data = msg.data
        return res
    
    def to_rosmsg(self):
        msg = Bool()
        msg.data = bool(self.data)
        return msg
    
    def to(self, device):
        self.device = device
        return self

    def to_kitti(self, base_dir, idx):
        """define how to convert this dtype to a kitti file
        """
        pass

    @classmethod
    def from_kitti(cls, base_dir, idx, device='cpu'):
        """define how to convert this dtype from a kitti file
        """
        pass

    def __repr__(self):
        return "BoolTorch with data {}, device {}".format(self.data, self.device)
