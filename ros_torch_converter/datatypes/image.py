import os
import cv2
import torch
import cv_bridge
import warnings
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from sensor_msgs.msg import Image, CompressedImage
from perception_interfaces.msg import FeatureImage

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class ImageTorch(TorchCoordinatorDataType):
    """
    TorchCoordinator class for images
    (note that this class is specifically an image type that can be cv_bridged),
    For arbitrary images, use FeatureImageTorch (which will serialize to a custom message instead)
    """
    to_rosmsg_type = Image
    from_rosmsg_type = Image

    def __init__(self, device):
        super().__init__()
        self.image = torch.zeros(0,0,3, device=device)
        self.bridge = cv_bridge.CvBridge()
        self.device = device

    def from_torch(image):
        if image.max() > 1.:
            warnings.warn("Found image with value > 1. Did you convert from uint8 to float?")

        res = ImageTorch(device=image.device)
        res.image = image.float()
        return res

    def from_numpy(image, device):
        if image.max() > 1.:
            warnings.warn("Found image with value > 1. Did you convert from uint8 to float?")

        res = ImageTorch(device=device)
        res.image = torch.tensor(image, dtype=torch.float32, device=device)
        return res

    def to_rosmsg(self, encoding='rgb8', compressed=False):
        img = (self.image*255.).cpu().numpy().astype(np.uint8)
        if compressed:
            img_msg = self.bridge.cv2_to_compressed_imgmsg(img, encoding=encoding)
        else:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)

        img_msg.header.stamp = time_to_stamp(self.stamp)
        img_msg.header.frame_id = self.frame_id
        return img_msg
    
    def from_rosmsg(msg, device='cpu'):
        res = ImageTorch(device)
        img = res.bridge.imgmsg_to_cv2(msg)[..., :3]
        img = torch.from_numpy(img/255.).float().to(device)
        res.image = img
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_kitti(self, base_dir, idx):
        save_fp = os.path.join(base_dir, "{:08d}.png".format(idx))
        img = (self.image * 255.).long().cpu().numpy()
        cv2.imwrite(save_fp, img)

    def from_kitti(self, base_dir, idx, device='cpu'):
        pass

    def to(self, device):
        self.device = device
        self.image = self.image.to(device)
        return self
    
    def __repr__(self):
        return "ImageTorch of shape {} (time = {:.2f}, frame = {}, device = {})".format(self.image.shape, self.stamp, self.frame_id, self.device)

class ThermalImageTorch(TorchCoordinatorDataType):
    """
    TorchCoordinator class for images
    (note that this class is specifically an image type that can be cv_bridged),
    For arbitrary images, use FeatureImageTorch (which will serialize to a custom message instead)
    """
    to_rosmsg_type = Image
    from_rosmsg_type = Image

    def __init__(self, device):
        super().__init__()
        self.image = torch.zeros(0,0,3, device=device)
        self.bridge = cv_bridge.CvBridge()
        self.device = device

    def from_torch(image):
        if image.max() > 1.:
            warnings.warn("Found image with value > 1. Did you convert from uint8 to float?")

        res = ThermalImageTorch(device=image.device)
        res.image = image.float()
        return res

    def from_numpy(image, device):
        if image.max() > 1.:
            warnings.warn("Found image with value > 1. Did you convert from uint8 to float?")

        res = ThermalImageTorch(device=device)
        res.image = torch.tensor(image, dtype=torch.float32, device=device)
        return res   

    def to_rosmsg(self, encoding='passthrough', compressed=False):
        img = (self.image*255.).cpu().numpy().astype(np.uint8)
        if compressed:
            img_msg = self.bridge.cv2_to_compressed_imgmsg(img, encoding=encoding)
        else:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)

        img_msg.header.stamp = time_to_stamp(self.stamp)
        img_msg.header.frame_id = self.frame_id
        return img_msg
    
    def from_rosmsg(msg, device='cpu'):
        '''
        Read 8-bit processed image from ROS message
        '''
        res = ThermalImageTorch(device)
        img = res.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        img = torch.from_numpy(img/255.).float().to(device)
        res.image = img
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_kitti(self, base_dir, idx):
        save_fp = os.path.join(base_dir, "{:08d}.png".format(idx))
        # img = (self.image * 255.).long().cpu().numpy() pdb
        # import pdb;pdb.set_trace()
        cv2.imwrite(save_fp, self.image)

    def from_kitti(self, base_dir, idx, device='cpu'):
        pass

    def to(self, device):
        self.device = device
        self.image = self.image.to(device)
        return self
    
    def __repr__(self):
        return "ThermalImageTorch of shape {} (time = {:.2f}, frame = {}, device = {})".format(self.image.shape, self.stamp, self.frame_id, self.device)
    
class FeatureImageTorch(TorchCoordinatorDataType):
    """
    TorchCoordinator class for feature images
    unlike ImageTorch, this class can take arbitrary image channels/features,
    and will serialze to perception_interfaces/FeatureImage instead of sensor_msgs/Image
    """
    to_rosmsg_type = FeatureImage
    from_rosmsg_type = FeatureImage

    def __init__(self, device):
        super().__init__()
        self.image = torch.zeros(0,0,3, device=device)
        self.device = device

    def from_torch(image):
        if image.dtype != torch.float32:
            warnings.warn('Got image type that isnt float32!')

        res = FeatureImageTorch(device=image.device)
        res.image = image.float()
        return res

    def from_numpy(image, device):
        if image.dtype != np.float32:
            warnings.warn('Got image type that isnt float32!')

        res = FeatureImageTorch(device=device)
        res.image = torch.tensor(image, dtype=torch.float32, device=device)
        return res
    
    def from_rosmsg(msg, device):
        warnings.warn('havent implemented featureimg->ros')
        return None

    def to_rosmsg(self):
        msg = FeatureImage()

        msg.height = self.image.shape[0]
        msg.width = self.image.shape[1]
        msg.num_channels = self.image.shape[2]

        msg.data = self.image.flatten().cpu().numpy().astype("f").tobytes()
        
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg

    def to_kitti(self, base_dir, idx):
        """define how to convert this dtype to a kitti file
        """
        pass

    def from_kitti(self, base_dir, idx, device):
        """define how to convert this dtype from a kitti file
        """
        pass

    def to(self, device):
        self.device = device
        self.image = self.image.to(device)
        return self
    
    def __repr__(self):
        return "FeatureImageTorch of shape {} (time = {:.2f}, frame = {}, device = {})".format(self.image.shape, self.stamp, self.frame_id, self.device)