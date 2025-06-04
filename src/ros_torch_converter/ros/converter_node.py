import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import yaml

from ros_torch_converter.converter import ROSTorchConverter

class ConverterNode(Node):
    """
    Dummy node for debugging converter
    """
    def __init__(self):
        super().__init__("converter_node")
        config_fp = self.declare_parameter(
            "config_fp", "/home/tartandriver/tartandriver_ws/src/core/torch_coordinator/config/debug_ros.yaml"
        ).value
        self.get_logger().info('reading config from {}'.format(config_fp))
        
        config = yaml.safe_load(open(config_fp, 'r'))
        config["ros_converter"]["device"] = config["common"]["device"]

        self.converter = ROSTorchConverter(config["ros_converter"], name="converter_node")

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        converter_status = self.converter.get_status_str()

        self.get_logger().info(
            "ROS cvt status:\n{}".format(converter_status),
        )

        if self.converter.can_get_data():
            data, times = self.converter.get_data(return_times=True)
            res = "\n"
            for k in data.keys():
                res += "{}: data = {}\n".format(k, str(data[k]))
            self.get_logger().info(res)

def main(args=None):
    rclpy.init(args=args)
    cvt_node = ConverterNode()
    executor = MultiThreadedExecutor()

    # Add both nodes to the executor
    executor.add_node(cvt_node)
    executor.add_node(cvt_node.converter)

    try:
        # Spin both nodes concurrently
        executor.spin()
    finally:
        # Clean up
        cvt_node.converter.destroy_node()
        cvt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
