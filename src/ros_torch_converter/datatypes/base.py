import abc

class TorchCoordinatorDataType(abc.ABC):
    """
    Interface class for TorchCoordinator datatypes (this is the only thing we want to support in the TorchCoordinator pipeline)
    #TODO think about integrating more cleanly into ros_torch_converter
    """
    def __init__(self):
        self.stamp = -1.
        self.frame_id = ""

    @property
    @abc.abstractmethod
    def to_rosmsg_type(self):
        """Define input rosmsg type
        """

    @property
    @abc.abstractmethod
    def from_rosmsg_type(self):
        """Define output rosmsg type
        """

    @abc.abstractmethod
    def to_rosmsg(self):
        """define how to convert this datatype to a ros message
        """
        pass

    @abc.abstractmethod
    def from_rosmsg(msg, device):
        """define how to convert this datatype from a ros message
        """
        pass

    @abc.abstractmethod
    def to_kitti(self, base_dir, idx):
        """define how to convert this dtype to a kitti file
        """
        pass

    @abc.abstractmethod
    def from_kitti(self, base_dir, idx, device):
        """define how to convert this dtype from a kitti file
        """
        pass

    @abc.abstractmethod
    def to(self, device):
        """define how to move this datatype to CPU/GPU
        """
        pass

    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda')