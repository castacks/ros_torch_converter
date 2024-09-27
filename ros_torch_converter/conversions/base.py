import abc

class Conversion(abc.ABC):
    """Base class for ROS->torch conversion. Defines how to go from message to type

    Current plan is to have one of these for every pair of msg->torch interface pairs

    Class naming convention should be <msgtype>To<torch_type> (e.g. GridMapToTorchMap)
    """
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def msg_type(self):
        """Declare the message type you are converting from
        """
        pass

    @abc.abstractmethod
    def cvt(self, msg):
        """Main convert method. This should take in a ROS message and output a torch tensor (or dict of torch tensors)
        """
        pass