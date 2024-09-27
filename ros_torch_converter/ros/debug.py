import yaml

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32

from ros_torch_converter.converter import ROSTorchConverter


def main(args=None):
    config_fp = "/home/tartandriver/tartandriver_ws/src/core/ros_torch_converter/ros_torch_converter/configs/costmap_speedmap.yaml"
    config = yaml.safe_load(open(config_fp, "r"))["ros"]

    rclpy.init(args=args)

    converter = ROSTorchConverter(config)

    rclpy.spin(converter)

    sub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
