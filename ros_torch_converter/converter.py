import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.float import Float32Torch
from ros_torch_converter.datatypes.image import ImageTorch, FeatureImageTorch, ThermalImageTorch, Thermal16bitImageTorch
from ros_torch_converter.datatypes.intrinsics import IntrinsicsTorch
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch, FeaturePointCloudTorch
from ros_torch_converter.datatypes.transform import TransformTorch, OdomTransformTorch
from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.goal_array import GoalArrayTorch

from tartandriver_utils.ros_utils import stamp_to_time

str_to_cvt_class = {
    "BEVGrid": BEVGridTorch,
    "Float32": Float32Torch,
    "Image": ImageTorch,
    "FeatureImage": FeatureImageTorch,
    "ThermalImage": ThermalImageTorch,
    "Thermal16bitImage": Thermal16bitImageTorch,
    "Intrinsics": IntrinsicsTorch,
    "PointCloud": PointCloudTorch,
    "FeaturePointCloud": FeaturePointCloudTorch,
    "Transform": TransformTorch,
    "OdomTransform": OdomTransformTorch,
    "OdomRBState": OdomRBStateTorch,
    "GoalArray": GoalArrayTorch
}

class ROSTorchConverter(Node):
    """Top-level class that manages conversion from ROS->torch.

    Essentially, this class will spin up a number of subscribers and store the latest message for each.
    When it is asked for data, it will convert all the messages to torch and return them as a (potentially nested) dict
    """

    def __init__(self, config, name=""):
        super().__init__(name + "_ros_torch_converter_node")

        self.config = config
        self.device = self.config['device']
        self.subscribers = {}
        self.converters = {}

        self.data = {}
        self.data_times = {}

        self.lock = False

        self.setup_subscribers()

        self.get_logger().info("cvt node ready")

    def setup_subscribers(self):
        #TODO: Setup an option using message_filters.TimeSynchronizer
        for topic_conf in self.config["topics"]:
            self.data[topic_conf["name"]] = None
            self.data_times[topic_conf["name"]] = -1.0

            self.converters[topic_conf["name"]] = str_to_cvt_class[topic_conf["type"]]

            sub = self.create_subscription(
                self.converters[topic_conf["name"]].from_rosmsg_type,  # Message type
                topic_conf["topic"],  # Topic name
                lambda msg, topic_conf=topic_conf: self.handle_msg(
                    msg, topic_conf
                ),  # Callback with additional args
                qos_profile=qos_profile_sensor_data,  # QoS (default queue size)
            )

    def handle_msg(self, msg, topic_conf):
        if not self.lock:
            self.data[topic_conf["name"]] = msg
            try:
                self.data_times[topic_conf["name"]] = stamp_to_time(msg.header.stamp)
            except:
                self.data_times[topic_conf["name"]] = stamp_to_time(self.get_clock().now().to_msg())

    def get_data(self, return_times=False, device="cpu"):
        self.lock = True
        data = {k: self.converters[k].from_rosmsg(msg, device=self.device) for k, msg in self.data.items()}
        times = copy.deepcopy(self.data_times)
        self.lock = False

        return (data, times) if return_times else data

    def can_get_data(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())

        for topic_config in self.config["topics"]:
            max_age = topic_config["max_age"]
            topic_name = topic_config["name"]
            data_time = self.data_times[topic_name]

            if curr_time - data_time > max_age or data_time < 0.:
                return False

        return True

    def get_status_str(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())
        out = "\n ---converter status--- \n"
        for topic_conf in self.config["topics"]:

            data_exists = self.data[topic_conf["name"]] is not None
            data_age = curr_time - self.data_times[topic_conf["name"]]
            out += "\t{:<16} exists: {} age:{:.2f}s\n".format(
                topic_conf["name"] + " " + topic_conf["topic"] + ":",
                data_exists,
                data_age,
            )

        out += "can get data: {}\n".format(self.can_get_data())
        out += "curr time: {}\n".format(curr_time)
        return out
