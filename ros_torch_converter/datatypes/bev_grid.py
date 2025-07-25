import os
import yaml
import copy
import torch
import array

import warnings
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType

from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap

from tartandriver_utils.ros_utils import time_to_stamp, stamp_to_time

import ros2_numpy_cpp

class BEVGridTorch(TorchCoordinatorDataType):
    """
    Wrapper around the BEVGrid class from visual mapping
    """
    to_rosmsg_type = GridMap
    from_rosmsg_type = GridMap

    def __init__(self, device):
        super().__init__()
        self.bev_grid = None
        self.device = device

    def from_bev_grid(bev_grid):
        res = BEVGridTorch(device=bev_grid.device)
        res.bev_grid = bev_grid
        return res
    
    def from_rosmsg(msg, feature_keys=[], device='cpu'):
        res = BEVGridTorch(device=device)
        orientation = np.array(
            [
                msg.info.pose.orientation.x,
                msg.info.pose.orientation.y,
                msg.info.pose.orientation.z,
                msg.info.pose.orientation.w,
            ]
        )
        assert np.allclose(
            orientation, np.array([0.0, 0.0, 0.0, 1.0])
        ), "ERROR: we dont support rotated gridmaps"

        if len(feature_keys) == 0:
            layers_to_extract = msg.layers
        else:
            layers_to_extract = [x for x in feature_keys if x in msg.layers]

            if len(layers_to_extract) != len(feature_keys):
                warnings.warn("warning: not all expected layers are in received gridmap!")

        metadata = LocalMapperMetadata(
            origin=[
                    msg.info.pose.position.x - 0.5 * msg.info.length_x,
                    msg.info.pose.position.y - 0.5 * msg.info.length_y,
                ],
            length=[msg.info.length_x, msg.info.length_y],
            resolution=[msg.info.resolution, msg.info.resolution],
        ).to(device)
        nx, ny = metadata.N.tolist()

        bev_grid_data = []

        for layer in layers_to_extract:
            idx = msg.layers.index(layer)
            data = np.array(msg.data[idx].data).reshape(nx, ny)[::-1, ::-1].copy().T
            data = torch.tensor(data)
            bev_grid_data.append(data)

        bev_grid_data = torch.stack(bev_grid_data, axis=0)

        bev_grid = BEVGrid(metadata=metadata, n_features=bev_grid_data.shape[0], feature_keys=layers_to_extract)
        bev_grid.data = bev_grid_data
        #TODO figure out something better than this
        bev_grid.known = torch.ones(nx, ny, device=device, dtype=torch.bool)

        res.bev_grid = bev_grid
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id

        return res

    def to_rosmsg(self, viz_features=True, viz_layers=['dino_0', 'dino_1', 'dino_2']):
        gridmap_msg = GridMap()

        gridmap_data = self.bev_grid.data.cpu().numpy()

        # setup metadata
        gridmap_msg.header.stamp = time_to_stamp(self.stamp)
        gridmap_msg.header.frame_id = self.frame_id

        gridmap_msg.layers = copy.deepcopy(self.bev_grid.feature_keys.label)

        has_unk = 'min_elevation_filtered_inflated_mask' in self.bev_grid.feature_keys.label

        if has_unk:
            unk_idx = self.bev_grid.feature_keys.index('min_elevation_filtered_inflated_mask')
            unk = np.copy(gridmap_data[..., unk_idx])
            unk[unk < 0.1] = float('nan')
            gridmap_data = np.concatenate([gridmap_data, np.expand_dims(unk,-1)], axis=-1)

            gridmap_msg.layers.append('viz_mask')
            gridmap_msg.basic_layers.append('viz_mask')

        gridmap_msg.info.resolution = self.bev_grid.metadata.resolution.mean().item()
        gridmap_msg.info.length_x = self.bev_grid.metadata.length[0].item()
        gridmap_msg.info.length_y = self.bev_grid.metadata.length[1].item()
        gridmap_msg.info.pose.position.x = (
            self.bev_grid.metadata.origin[0].item() + 0.5 * gridmap_msg.info.length_x
        )
        gridmap_msg.info.pose.position.y = (
            self.bev_grid.metadata.origin[1].item() + 0.5 * gridmap_msg.info.length_y
        )
        gridmap_msg.info.pose.position.z = 0.
        gridmap_msg.info.pose.orientation.w = 1.0
        # transposed_layer_data = np.transpose(gridmap_data, (0, 2,1))
        # flipped_layer_data = np.flip(np.flip(transposed_layer_data, axis=1), axis=2)

        # gridmap_data has the shape (rows, cols, layers)
        # Step 1: Flip the 2D grid layers in both directions (reverse both axes)
        flipped_data = np.flip(gridmap_data, axis=(0, 1))  # Flips along both axes

        # Step 2: Transpose the first two dimensions (x, y) for each layer
        transposed_data = np.transpose(
            flipped_data, axes=(1, 0, 2)
        )  # Transpose rows and cols

        # Step 3: Flatten each 2D layer, maintaining the layers' structure (flattening across x, y)
        flattened_data = transposed_data.reshape(-1, gridmap_data.shape[-1])

        bev_size = torch.prod(self.bev_grid.metadata.N).item()

        accum_time = 0
        for i in range(gridmap_data.shape[-1]):
            layer_data = gridmap_data[..., i]
            gridmap_layer_msg = Float32MultiArray()
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="column_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0],
                )
            )
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0] * layer_data.shape[1],
                )
            )

            # # 1. slow tolist()
            # gridmap_layer_msg.data = flattened_data[:, i].tolist()

            gridmap_layer_msg.data = array.array('f', [0] * bev_size)
            ros2_numpy_cpp.npcpp_deepcopy(
                np.ascontiguousarray(flattened_data[:, i]),
                gridmap_layer_msg.data.buffer_info()[0],
                flattened_data[:, i].shape[0]
            )            

            gridmap_msg.data.append(gridmap_layer_msg)

        # add dummy elevation
        gridmap_msg.layers.append("elevation")
        layer_data = (
            np.zeros_like(gridmap_data[..., 0])
        )
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0],
            )
        )
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0] * layer_data.shape[1],
            )
        )

        # 2. slow tolist()
        # gridmap_layer_msg.data = layer_data.flatten().tolist()
        gridmap_layer_msg.data = array.array('f', [0] * bev_size)
        tmp = layer_data.flatten()
        ros2_numpy_cpp.npcpp_deepcopy(
            np.ascontiguousarray(tmp),
            gridmap_layer_msg.data.buffer_info()[0],
            tmp.shape[0]
        )                    
        gridmap_msg.data.append(gridmap_layer_msg)

        # TODO: figure out how to support multiple viz output types
        if gridmap_data.shape[-1] > 2 and (viz_layers[0] in self.bev_grid.feature_keys.label):
            viz_idxs = [self.bev_grid.feature_keys.label.index(k) for k in viz_layers]
            gridmap_rgb = gridmap_data[..., viz_idxs]
            vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1, 1, 3)
            vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)

            gridmap_cs = ((gridmap_rgb - vmin) / (vmax - vmin)).clip(0.0, 1.0)
            gridmap_cs = (gridmap_cs * 255.0).astype(np.int32)

            gridmap_color = (
                gridmap_cs[..., 0] * (2 ** 16)
                + gridmap_cs[..., 1] * (2 ** 8)
                + gridmap_cs[..., 2]
            )
            gridmap_color = gridmap_color.view(dtype=np.float32)

            gridmap_msg.layers.append("rgb_viz")
            gridmap_layer_msg = Float32MultiArray()
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="column_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0],
                )
            )
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0] * layer_data.shape[1],
                )
            )

            # # 3. slow tolist()
            # gridmap_layer_msg.data = gridmap_color.T[::-1, ::-1].flatten().tolist()
            gridmap_layer_msg.data = array.array('f', [0] * bev_size)
            tmp = gridmap_color.T[::-1, ::-1].flatten()
            ros2_numpy_cpp.npcpp_deepcopy(
                np.ascontiguousarray(tmp),
                gridmap_layer_msg.data.buffer_info()[0],
                tmp.shape[0]
            )                                
            gridmap_msg.data.append(gridmap_layer_msg)

        gridmap_msg.header.stamp = time_to_stamp(self.stamp)
        gridmap_msg.header.frame_id = self.frame_id

        return gridmap_msg

    def to_kitti(self, base_dir, idx):
        data_fp = os.path.join(base_dir, "{:08d}_data.npy".format(idx))
        metadata_fp = os.path.join(base_dir, "{:08d}_metadata.yaml".format(idx))

        metadata = {
            'feature_keys': [
                f"{label}, {meta}" for label, meta in zip(
                    self.bev_grid.feature_keys.label,
                    self.bev_grid.feature_keys.metainfo
                )
            ],
            'length': self.bev_grid.metadata.length.cpu().numpy().tolist(),
            'origin': self.bev_grid.metadata.origin.cpu().numpy().tolist(),
            'resolution': self.bev_grid.metadata.resolution.cpu().numpy().tolist()
        }

        yaml.dump(metadata, open(metadata_fp, 'w'), default_flow_style=False)

        data = self.bev_grid.data.cpu().numpy()
        np.save(data_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        data_fp = os.path.join(base_dir, "{:08d}_data.npy".format(idx))
        metadata_fp = os.path.join(base_dir, "{:08d}_metadata.yaml".format(idx))

        metadata = yaml.safe_load(open(metadata_fp))
        labels, metas = zip(*[s.split(', ') for s in metadata['feature_keys']])
        feature_keys = FeatureKeyList(label=list(labels), metadata=list(metas))
        
        metadata = LocalMapperMetadata(
            origin = metadata['origin'],
            length = metadata['length'],
            resolution = metadata['resolution'],
            device = device
        )

        bev_grid = BEVGrid(
            metadata = metadata,
            n_features = len(feature_keys),
            feature_keys = feature_keys,
        )
        
        data = np.load(data_fp)
        bev_grid.data = torch.tensor(data, dtype=torch.float, device=device)
        
        gt = BEVGridTorch.from_bev_grid(bev_grid)
        return gt
    
    def to(self, device):
        self.device = device
        self.bev_grid = self.bev_grid.to(device)
        return self
    
    def __repr__(self):
        return "BEVGridTorch of size {}, time = {:.2f}, frame = {}, device = {} features = {}".format(self.bev_grid.data.shape, self.stamp, self.frame_id, self.device, self.bev_grid.feature_keys)