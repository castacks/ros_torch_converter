# ros_torch_converter

This package provides a relatively flexible way of handling topic conversion/subscription for ROS nodes that need a large amount of data converted to torch. 

NOTE: The datatypes for this package are currently defined in ```torch_coordinator```.

## Usage

New to ROS2, classes need to be a Node in order to instantiate subscribers, etc. Thus, the current way to use this converter is something like the following:

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ros_torch_converter.converter import ROSTorchConverter

class ExampleNode(Node):
    def __init__(self, converter):
        self.converter = converter
        self.timer = self.get_timer(1.0, self.callback)

    def callback(self):
        if self.converter.can_get_data():
            #dict of torch data
            data = self.converter.get_data() 

if __name__ == '__main__':
    converter_node = ROSTorchConverter(config)
    plt_node = MPLPlotter(converter_node)

    executor = MultiThreadedExecutor()
    executor.add_node(converter_node)
    executor.add_node(plt_node)

    try:
        executor.spin()
    finally:
        converter_node.destroy_node()
        plt_node.destroy_node()
        rclpy.shutdown()
```

The ROS-Torch converter requires a config yaml (example in `configs/costmap_speedmap.yaml`) that specifies the topics to listen to and how to convert.