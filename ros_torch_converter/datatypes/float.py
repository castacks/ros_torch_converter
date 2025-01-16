import torch

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from std_msgs.msg import Float32

class Float32Torch(TorchCoordinatorDataType):
    """
    """
    to_rosmsg_type = Float32
    from_rosmsg_type = Float32

    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        self.data = torch.zeros(1, device=device)
        self.device = device
    
    def from_rosmsg(msg, device):
        res = Float32Torch(device=device)
        res.data = torch.tensor([msg.data], device=device)
        return res
    
    def to_rosmsg(self):
        msg = Float32()
        msg.data = self.data.item()
        return msg
    
    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def __repr__(self):
        return "Float32Torch with data {:.2f}, device {}".format(self.data.item(), self.device)