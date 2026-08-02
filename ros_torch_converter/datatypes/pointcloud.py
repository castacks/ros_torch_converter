import os
import yaml
import array
import torch
import rosbags
import warnings
import ros2_numpy
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

from ros_torch_converter.datatypes.feature_key_list import FeatureKeyList

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import update_frame_file, update_timestamp_file, read_frame_file, read_timestamp_file

class PointCloudTorch(TorchCoordinatorDataType):
    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, device):
        super().__init__()
        self.pts = torch.zeros(0, 3, device=device)
        self.colors = torch.zeros(0, 3, device=device)
        self.feature_keys = FeatureKeyList(
            label=['x', 'y', 'z'],
            metainfo=['raw'] * 3
        )
        self.device = device

    def clone(self):
        out = PointCloudTorch(device=self.device)
        out.pts = self.pts.clone()
        out.colors = self.colors.clone()
        out.stamp = self.stamp
        out.frame_id = self.frame_id
        return out

    def apply_mask(self, mask):
        mask_pts = self.pts[mask]
        mask_colors = self.colors[mask]
        return PointCloudTorch.from_torch(
            pts=mask_pts,
            colors=mask_colors
        )
    
    def from_rosmsg(msg, device='cpu'):
        #HACK to get ros2_numpy to cooperate with rosbags dtypes.
        #TODO write a script to do this for all types
        if type(msg) != PointCloud2:
            k = (PointCloud2, False)
            k2 = (type(msg), False)
            ros2_numpy.registry._to_numpy[k2] = ros2_numpy.registry._to_numpy[k]

        res = PointCloudTorch(device=device)
        pcl_np = ros2_numpy.numpify(msg)
        xyz = pcl_np['xyz']

        res.pts = torch.tensor(xyz).float().to(res.device)

        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def from_numpy(pts, colors=None, device='cpu'):
        res = PointCloudTorch(device=device)
        res.pts = torch.tensor(pts, dtype=torch.float32, device=device)
        if colors is not None:
            res.colors = torch.tensor(colors, dtype=torch.float32, device=device)

        return res

    def from_torch(pts, colors=None):
        res = PointCloudTorch(device=pts.device)
        res.pts = pts.float()
        if colors is not None:
            res.colors = colors.float()
        return res

    def to_rosmsg(self):
        points = self.pts.cpu().numpy().astype(np.float32)
        colors = self.colors.cpu().numpy()

        msg = PointCloud2()
        msg.height = 1
        msg.width = points.shape[0]
        msg.point_step = 12

        msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=msg.width) for i,n in enumerate('xyz')]

        data = points

        if self.colors.shape[0] > 0:
            msg.point_step += 4
            msg.fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=msg.width))

            r = (colors[:, 0] * 255).astype(np.uint32)
            g = (colors[:, 1] * 255).astype(np.uint32)
            b = (colors[:, 2] * 255).astype(np.uint32)
            rgb = (r<<16) | (g<<8) | (b<<0)
            rgb.dtype = np.float32

            data = np.concatenate([data, rgb.reshape(-1, 1)], axis=-1)

        data = data.flatten()

        # borrowing from https://github.com/Box-Robotics/ros2_numpy/blob/humble/ros2_numpy/point_cloud2.py
        mem_view = memoryview(data)

        if mem_view.nbytes > 0:
            array_bytes = mem_view.cast("B")
        else:
            array_bytes = b""

        as_array = array.array("B")
        as_array.frombytes(array_bytes)

        msg.data = as_array
            
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id
        return msg

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, "{:08d}.npy".format(idx))

        if len(self.pts) == len(self.colors):
            pts = torch.cat([self.pts, self.colors], dim=-1).cpu().numpy()
        else:
            pts = self.pts.cpu().numpy()

        np.save(save_fp, pts)

    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "{:08d}.npy".format(idx))

        pts = np.load(fp)
        pc = PointCloudTorch(device=device)
        pc.pts = torch.tensor(pts[:, :3], device=device).float()

        if pts.shape[-1] == 6:
            pc.colors = torch.tensor(pts[:, 3:6], device=device).float()

        pc.stamp = read_timestamp_file(base_dir, idx)
        pc.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return pc

    def rand_init(device='cpu'):
        pts = torch.rand(10000, 3, device=device)
        colors = torch.rand(10000, 3, device=device)

        pct = PointCloudTorch.from_torch(pts, colors)
        pct.frame_id = 'random'
        pct.stamp = np.random.rand()

        return pct

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.pts, other.pts):
            return False

        if not torch.allclose(self.colors, other.colors):
            return False

        return True

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

    Major difference is that not all points need have a feature,
        so maintain an additional mask, such that pts[mask] = features
    """
    to_rosmsg_type = PointCloud2
    from_rosmsg_type = PointCloud2

    def __init__(self, feature_keys, device):
        super().__init__()
        self.pts = torch.zeros(0, 3, device=device)
        self.features = torch.zeros(0, 0, device=device)
        self.feat_mask = torch.zeros(0, device=device, dtype=torch.bool)
        self.feature_keys = feature_keys
        self.device = device

    def clone(self):
        out = FeaturePointCloudTorch(device=self.device)
        out.pts = self.pts.clone()
        out.features = self.features.clone()
        out.feat_mask = self.feat_mask.clone()
        out.stamp = self.stamp
        out.frame_id = self.frame_id
        return out

    def apply_mask(self, mask):
        """
        Args:
            mask: [N] bool tensor where N=num pts 
        Returns: 
            FeaturePointCloudTorch with masked points
        """
        mask_pts = self.pts[mask]
        mask_feat_mask = self.feat_mask[mask]
        mask_feats = self.features[mask[self.feat_mask]]

        return FeaturePointCloudTorch.from_torch(
            pts=mask_pts,
            features=mask_feats,
            mask=mask_feat_mask,
            feature_keys=self.feature_keys
        )

    @property
    def feature_pts(self):
        return self.pts[self.feat_mask]

    @property
    def non_feature_pts(self):
        return self.pts[~self.feat_mask]

    def from_rosmsg(msg, device):
        warnings.warn('havent implemented ros->featpc yet')
        res = FeaturePointCloudTorch(device=device)
        return res

    def from_torch(pts, features, mask, feature_keys):
        assert len(mask) == len(pts), "expected len(mask) == len(pts)"
        assert mask.sum() == len(features), "expected mask.sum() == len(features)"
        assert features.shape[-1] == len(feature_keys), "expected features.shape[-1] == len(feature_keys)"

        res = FeaturePointCloudTorch(feature_keys=feature_keys, device=pts.device)
        res.pts = pts.float()
        res.features = features.float()
        res.feat_mask = mask.bool()
        return res
    
    def from_numpy(pts, features, mask, feature_keys, device):
        assert len(mask) == len(pts), "expected len(mask) == len(pts)"
        assert mask.sum() == len(features), "expected mask.sum() == len(features)"
        assert feature.shape[-1] == len(feature_keys), "expected feature.shape[-1] == len(feature_keys)"

        res = FeaturePointCloudTorch(feature_keys=feature_keys, device=device)
        res.pts = torch.from_numpy(pts, dtype=torch.float32, device=device)
        res.features = torch.from_numpy(features, dtype=torch.float32, device=device)
        res.feat_mask = torch.from_numpy(mask, dtype=torch.bool, device=device)
        return res
    
    def to_rosmsg(self):
        warnings.warn('havent implemented featpc->ros message yet')
        pass

    def to_kitti(self, base_dir, idx):
        """define how to convert this dtype to a kitti file
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)
        
        data_fp = os.path.join(base_dir, "{:08d}_data.npz".format(idx))
        metadata_fp = os.path.join(base_dir, "{:08d}_metadata.yaml".format(idx))

        metadata = {
            'feature_keys': [
                f"{label}, {meta}" for label, meta in zip(
                    self.feature_keys.label,
                    self.feature_keys.metainfo
                )
            ]
        }

        yaml.dump(metadata, open(metadata_fp, 'w'))

        data = {
            'pts': self.pts.cpu().numpy(),
            'features': self.features.cpu().numpy(),
            'feat_mask': self.feat_mask.cpu().numpy()
        } 
        np.savez(data_fp, **data)

    def from_kitti(base_dir, idx, device='cpu'):
        """define how to convert this dtype from a kitti file
        """
        metadata_fp = os.path.join(base_dir, "{:08d}_metadata.yaml".format(idx))
        metadata = yaml.safe_load(open(metadata_fp, 'r'))

        labels, metas = zip(*[s.split(', ') for s in metadata['feature_keys']])
        feature_keys = FeatureKeyList(label=list(labels), metainfo=list(metas))
        
        data_fp = os.path.join(base_dir, "{:08d}_data.npz".format(idx))
        pc_data = np.load(data_fp)

        pts = torch.tensor(pc_data['pts'], dtype=torch.float, device=device)
        features = torch.tensor(pc_data['features'], dtype=torch.float, device=device)
        mask = torch.tensor(pc_data['feat_mask'], dtype=torch.bool, device=device)

        fpct = FeaturePointCloudTorch.from_torch(pts=pts, features=features, mask=mask, feature_keys=feature_keys)

        fpct.stamp = read_timestamp_file(base_dir, idx)
        fpct.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return fpct

    def to(self, device):
        self.device=device
        self.pts = self.pts.to(device)
        self.features = self.features.to(device)
        return self
    
    def rand_init(device='cpu'):
        pts = torch.rand(10000, 3, device=device)
        feats = torch.randn(5000, 5, device=device)
        idxs = torch.randperm(10000)[:5000]
        mask = torch.zeros(10000, dtype=torch.bool, device=device)
        mask[idxs] = True

        fks = FeatureKeyList(
            label = [f"feat_{i}" for i in range(5)],
            metainfo = ["rand"] * 5
        )

        fpct = FeaturePointCloudTorch.from_torch(pts=pts, features=feats, mask=mask, feature_keys=fks)
        fpct.frame_id = 'random'
        fpct.stamp = np.random.rand()
        return fpct

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.pts, other.pts):
            return False

        if not torch.allclose(self.features, other.features):
            return False

        if not (self.feat_mask == other.feat_mask).all():
            return False

        return True

    def __repr__(self):
        return "FeaturePointCloudTorch of shape {} (feats: {}, feat_size: {}, time = {:.2f}, frame_id = {}, device = {})".format(self.pts.shape, self.feature_keys, self.features.shape, self.stamp, self.frame_id, self.device)
