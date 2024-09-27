import torch
import numpy as np

from grid_map_msgs.msg import GridMap

from ros_torch_converter.conversions.base import Conversion


class GridMapToTorchMap(Conversion):
    """Convert a GridMap to a torch gridmap

    A torch GridMap is a dictionary with the following:
    metadata:
        length: [2] Tensor containing length in x,y
        origin: [2] Tensor of the coordinates of the lower-left cell
        resolution: [2] Tensor of the x/y spatial extent of a cell
    feature_keys: A [C] element list describing the c-th channel of data
    data: A [C x W x H] Tensor of actual gridmap data
    """

    def __init__(self, feature_keys=[]):
        """
        Args:
            feature_keys: a list of feature keys to extract from the message (everything by default)
        """
        self.feature_keys = feature_keys

    @property
    def msg_type(self):
        return GridMap

    def cvt(self, msg):
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

        if len(self.feature_keys) == 0:
            layers_to_extract = msg.layers
        else:
            layers_to_extract = [x for x in msg.layers if x in self.feature_keys]

        if len(layers_to_extract) != len(self.feature_keys):
            print("warning: not all expected layers are in received gridmap!")

        metadata = {
            "origin": torch.tensor(
                [
                    msg.info.pose.position.x - 0.5 * msg.info.length_x,
                    msg.info.pose.position.y - 0.5 * msg.info.length_y,
                ]
            ),
            "length": torch.tensor([msg.info.length_x, msg.info.length_y]),
            "resolution": torch.tensor([msg.info.resolution, msg.info.resolution]),
        }

        nx = round(msg.info.length_x / msg.info.resolution)
        ny = round(msg.info.length_y / msg.info.resolution)
        res = []

        for layer in layers_to_extract:
            idx = msg.layers.index(layer)
            data = np.array(msg.data[idx].data).reshape(nx, ny)[::-1, ::-1].copy().T
            data = torch.tensor(data)
            res.append(data)

        res = torch.stack(res, axis=0)

        return {"data": res, "metadata": metadata, "feature_keys": layers_to_extract}
