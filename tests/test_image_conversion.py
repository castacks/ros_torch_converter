import cv2
import numpy as np
import torch
from cv_bridge import CvBridge

from ros_torch_converter.datatypes.image import ImageTorch


def test_image_ros_to_kitti_stays_float32(tmp_path):
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    msg = CvBridge().cv2_to_imgmsg(pixels, encoding="bgr8")

    image = ImageTorch.from_rosmsg(msg)
    image.to_kitti(tmp_path, 0)

    assert image.image.dtype == torch.float32
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "00000000.png")), pixels)
