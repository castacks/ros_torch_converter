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

Extract kitti format data from rosbag:
```
python3 scripts/ros2bag_2_kitti.py --config config/kitti_config/super_odometry_sensors.yaml --src_dir [mcap_dir] --dst_dir [output_dir]
```

Export full scene reconstruction from SLAM to PCD:
```
python3 scripts/slam_2_pcd.py --config config/kitti_config/slam_2_pcd.yaml
```

Post-process SLAM dataset (depth, camera poses):
```
python3 scripts/get_slam_data.py --dataset [dataset_dir] --depth --odom \
  --config config/kitti_config/get_slam_data_rgb.yaml config/kitti_config/get_slam_data_thermal.yaml
```
- `--depth` extract depth maps, `--odom` extract camera poses 
- `--config` accepts multiple configs to process RGB + thermal in one run
- `--idx N` debug extracting single frame, `--idx N M` debug extracting a range
- `--verbose` per-frame printouts, `--viz` debug visualizations
- `--resume` resume from last processed frame, `--seq_to N` process first N frames only

Post-process thermal data (rectify, process):
```
python3 scripts/get_thermal_data.py --dataset [dataset_dir] --config config/kitti_config/get_thermal_data.yaml
```
- run this if your bag doesn't already have _rect or _processed thermal topics 

More details on [Slite](https://airlab.slite.com/app/docs/FAtc3skQoXn3r_).

### Transforms

ros_torch_converter uses the TfManager to handle transforms between different frames. It can get very confusing with different conventions. See [demo_tf_transform.py](scripts/demo_tf_transform.py) for a clear example that shows how to extract any extrinsics in the tf tree.

`tf_manager.get_transform(frame_a, frame_b, timestamp)` returns the pose of frame_b expressed in frame_a, `T_{a<-b}` that maps point from frame_b into frame_a.

## Testing
When you make changes to the datatypes, make sure this test passes:
```
python3 -m pytest tests/test.py 
```