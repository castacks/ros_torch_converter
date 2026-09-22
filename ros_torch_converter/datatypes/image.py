import array
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
        # Spot hand JPEG is published as bgr8 (OpenCV imdecode). Store RGB so
        # last-write voxel channels labeled r,g,b match the camera and RViz.
        encoding = (getattr(msg, "encoding", "") or "").split(";")[0].strip().lower()
        if encoding in ("bgr8", "bgra8", "rgb8", "rgba8", "8uc3", "8uc4"):
            img = res.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        else:
            img = res.bridge.imgmsg_to_cv2(msg)
        img = img[..., :3] if getattr(img, "ndim", 0) == 3 else img
        img = torch.from_numpy(img / 255.0).float().to(device)
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

class CompressedImageTorch(ImageTorch):
    """JPEG sibling of ImageTorch, for viz topics that cross the radio link.

    The full-resolution hand frame is ~900 KB as raw rgb8 and the Domo Tactical
    link preferentially drops large samples, so the operator-context crop is
    published a second time as JPEG. This is a separate datatype rather than a
    flag on ImageTorch because the coordinator's ros_hooks bind one message type
    per data key: the compressed topic is an additional key alongside the raw
    one, so the raw topic, offline RViz and bag recording are left unchanged.
    """
    to_rosmsg_type = CompressedImage
    from_rosmsg_type = CompressedImage

    #: cv2 JPEG quality (0-100). 80 keeps the bbox and scene legible at roughly
    #: a twelfth of the raw size.
    jpeg_quality = 80

    def from_torch(image):
        if image.max() > 1.:
            warnings.warn("Found image with value > 1. Did you convert from uint8 to float?")

        res = CompressedImageTorch(device=image.device)
        res.image = image.float()
        return res

    def from_numpy(image, device):
        if image.max() > 1.:
            warnings.warn("Found image with value > 1. Did you convert from uint8 to float?")

        res = CompressedImageTorch(device=device)
        res.image = torch.tensor(image, dtype=torch.float32, device=device)
        return res

    def to_rosmsg(self):
        img = (self.image * 255.).cpu().numpy().astype(np.uint8)
        # cv2.imencode expects BGR; self.image is RGB. Converting here rather
        # than relying on the caller keeps the channel order in one place.
        ok, buf = cv2.imencode(
            '.jpg',
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed for image of shape {}".format(img.shape))

        img_msg = CompressedImage()
        img_msg.format = 'jpeg'
        img_msg.data = array.array('B', buf.tobytes())
        img_msg.header.stamp = time_to_stamp(self.stamp)
        img_msg.header.frame_id = self.frame_id
        return img_msg

    def from_rosmsg(msg, device='cpu'):
        res = CompressedImageTorch(device)
        # imdecode returns BGR; store RGB to match ImageTorch.
        img = res.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = img[..., :3] if getattr(img, "ndim", 0) == 3 else img
        res.image = torch.from_numpy(img / 255.0).float().to(device)
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def __repr__(self):
        return "CompressedImageTorch of shape {} (time = {:.2f}, frame = {}, device = {})".format(
            self.image.shape, self.stamp, self.frame_id, self.device
        )


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
        img = np.stack([img] * 3, axis=-1)
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
        return "ThermalImageTorch of shape {} (time = {:.2f}, frame = {}, device = {})".format(self.image.shape, self.stamp, self.frame_id, self.device)

class Thermal16bitImageTorch(TorchCoordinatorDataType):
     """
     Specifically for 16-bit [0-65535] raw thermal data.
     Do not change to 8-bit [0-255] range.
     """
     to_rosmsg_type = Image
     from_rosmsg_type = Image
 
     def __init__(self, device):
         super().__init__()
         self.image = torch.zeros(0,0,3, device=device)
         self.bridge = cv_bridge.CvBridge()
         self.device = device
 
     def from_torch(image):
         res = Thermal16bitImageTorch(device=image.device)
         res.image = image.float()
         return res
 
     def from_numpy(image, device):
         res = Thermal16bitImageTorch(device=device)
         res.image = torch.tensor(image, dtype=torch.float32, device=device)
         return res   
 
     def to_rosmsg(self, encoding='passthrough', compressed=False):
         '''
         Convert 16-bit raw image to ROS message
         '''
         img = (self.image).cpu().numpy().astype(np.uint16)
         if compressed:
             img_msg = self.bridge.cv2_to_compressed_imgmsg(img, encoding=encoding)
         else:
             img_msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
 
         img_msg.header.stamp = time_to_stamp(self.stamp)
         img_msg.header.frame_id = self.frame_id
         return img_msg
     
     def from_rosmsg(msg, device='cpu'):
         '''Read 16-bit raw image from ROS message. Return as torch tensor.'''
         res = Thermal16bitImageTorch(device)
         img = res.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
         img = torch.from_numpy(img.astype(np.float32)).to(device)
         res.image = img
         res.stamp = stamp_to_time(msg.header.stamp)
         res.frame_id = msg.header.frame_id
         return res
 
     def to_kitti(self, base_dir, idx):
         '''Save 16-bit raw image'''
         save_fp = os.path.join(base_dir, "{:08d}.png".format(idx))
         if self.image.device.type != 'cpu':
             img_np = self.image.cpu().numpy().astype(np.uint16)
         else:
             img_np = self.image.numpy().astype(np.uint16)
         cv2.imwrite(save_fp, img_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])
 
     def from_kitti(self, base_dir, idx, device='cpu'):
         pass
 
     def to(self, device):
         self.device = device
         self.image = self.image.to(device)
         return self
     
     def __repr__(self):
         return "Thermal16bitImageTorch of shape {} (time = {:.2f}, frame = {}, device = {})".format(self.image.shape, self.stamp, self.frame_id, self.device)
     
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
        self.feature_keys = None
        self.device = device

    def from_torch(image, feature_keys=None):
        if image.dtype != torch.float32:
            warnings.warn('Got image type that isnt float32!')

        res = FeatureImageTorch(device=image.device)
        res.image = image.float()
        res.feature_keys = feature_keys
        return res

    def from_numpy(image, device):
        if image.dtype != np.float32:
            warnings.warn('Got image type that isnt float32!')

        res = FeatureImageTorch(device=device)
        res.image = torch.tensor(image, dtype=torch.float32, device=device)
        res.feature_keys = None
        return res
    
    def from_rosmsg(msg, device):
        warnings.warn('havent implemented featureimg->ros')
        return None

    def to_rosmsg(self):
        msg = FeatureImage()

        msg.height = self.image.shape[0]
        msg.width = self.image.shape[1]
        msg.num_channels = self.image.shape[2]

        # array.array, not bytes: FeatureImage.data is a uint8[] field and rclpy
        # converts a bytes object element-wise in Python (~48 ms per MB), while an
        # array.array is taken directly for a byte-identical payload.
        msg.data = array.array(
            "B", self.image.flatten().cpu().numpy().astype("f").tobytes()
        )
        
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
