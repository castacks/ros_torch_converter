import torch
import ros2_numpy
import warnings
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from sensor_msgs.msg import PointCloud2

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class PointCloudTorch(TorchCoordinatorDataType):
    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, device):
        super().__init__()
        self.pts = torch.zeros(0, 3, device=device)
        self.colors = torch.zeros(0, 3, device=device)
        self.device = device
    
    def from_rosmsg(msg, device):
        res = PointCloudTorch(device=device)
        pcl_np = ros2_numpy.numpify(msg)
        xyz = np.stack(
            [pcl_np["x"].flatten(), pcl_np["y"].flatten(), pcl_np["z"].flatten()], axis=-1
        )

        #TODO rgb

        res.pts = torch.from_numpy(xyz).float().to(res.device)

        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(pts, colors=None, device='cpu'):
        res = PointCloudTorch(device=device)
        res.pts = torch.from_numpy(pts, dtype=torch.float32, device=device)
        if colors is not None:
            res.colors = torch.from_numpy(colors, dtype=torch.float32, device=device)

        return res

    def from_torch(pts, colors=None):
        res = PointCloudTorch(device=pts.device)
        res.pts = pts.float()
        if colors is not None:
            res.colors = colors.float()
        return res

    def to_rosmsg(self):
        points = self.pts.cpu().numpy()
        if self.colors.shape[0] > 0:
            rgb_values = (self.colors * 255.0).cpu().numpy().astype(np.uint8)
            # Prepare the data array with XYZ and RGB
            xyzcolor = np.zeros(
                points.shape[0],
                dtype=[
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("rgb", np.float32),
                ],
            )

            # Assign XYZ values
            xyzcolor["x"] = points[:, 0]
            xyzcolor["y"] = points[:, 1]
            xyzcolor["z"] = points[:, 2]

            color = np.zeros(
                points.shape[0], dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8)]
            )
            color["r"] = rgb_values[:, 0]
            color["g"] = rgb_values[:, 1]
            color["b"] = rgb_values[:, 2]
            xyzcolor["rgb"] = ros2_numpy.point_cloud2.merge_rgb_fields(color)

            msg = ros2_numpy.msgify(PointCloud2, xyzcolor)
        else:
            xyzcolor = np.zeros(
                points.shape[0],
                dtype=[
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                ],
            )

            # Assign XYZ values
            xyzcolor["x"] = points[:, 0]
            xyzcolor["y"] = points[:, 1]
            xyzcolor["z"] = points[:, 2]

            msg = ros2_numpy.msgify(PointCloud2, xyzcolor)

        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg

    def to(self, device):
        self.device = device
        self.pts = self.pts.to(device)
        self.colors = self.colors.to(device)
        return self
        
    def __repr__(self):
        return "PointCloudTorch of shape {} (color: {}, time = {:.2f}, frame_id = {}, device = {})".format(self.pts.shape, self.colors.shape, self.stamp, self.frame_id, self.device)

class FeaturePointCloudTorch(TorchCoordinatorDataType):
    """
    PointCloud with an abritrary set of point features
    """
    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, device):
        self.pts = torch.zeros(0, 3, device=device)
        self.features = torch.zeros(0, 0, device=device)
        self.device = device

    def from_rosmsg(msg, device):
        warnings.warn('havent implemented ros->featpc yet')
        res = FeaturePointCloudTorch(device=device)
        return res

    def from_torch(pts, features):
        res = FeaturePointCloudTorch(device=pts.device)
        res.pts = pts.float()
        res.features = features.float()
        return res
    
    def from_numpy(pts, features, device):
        res = FeaturePointCloudTorch(device=device)
        res.pts = torch.from_numpy(pts, dtype=torch.float32, device=device)
        res.features = torch.from_numpy(features, dtype=torch.float32, device=device)
        return res
    
    def to_rosmsg(self):
        warnings.warn('havent implemented featpc->ros message yet')
        pass

    def to(self, device):
        self.device=device
        self.pts = self.pts.to(device)
        self.features = self.features.to(device)
        return self
    

    def __repr__(self):
        return "FeaturePointCloudTorch of shape {} (feats: {}, time = {:.2f}, frame_id = {}, device = {})".format(self.pts.shape, self.features.shape, self.stamp, self.frame_id, self.device)
