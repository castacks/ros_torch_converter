import rclpy
from rclpy.node import Node

from ros_torch_converter.conversions.float32 import Float32ToFloatTensor

from ros_torch_converter.conversions.posearray import PoseArrayTo2DGoalArray

from ros_torch_converter.conversions.odometry import OdometryToPoseTwist, OdometryToPose

from ros_torch_converter.conversions.gridmap import GridMapToTorchMap

str_to_cvt_class = {
    "Float32ToFloatTensor": Float32ToFloatTensor,
    "OdometryToPoseTwist": OdometryToPoseTwist,
    "OdometryToPose": OdometryToPose,
    "PoseArrayTo2DGoalArray": PoseArrayTo2DGoalArray,
    "GridMapToTorchMap": GridMapToTorchMap,
}


def get_ns_in_sec():  # TODO: move to utils
    return (
        rclpy.clock.Clock().now().to_msg().sec
        + rclpy.clock.Clock().now().to_msg().nanosec * 1e-9
    )


class ROSTorchConverter(Node):
    """Top-level class that manages conversion from ROS->torch.

    Essentially, this class will spin up a number of subscribers and store the latest message for each.
    When it is asked for data, it will convert all the messages to torch and return them as a (potentially nested) dict
    """

    def __init__(self, config):
        super().__init__("ros_torch_converter_node", use_global_arguments=False)

        self.config = config
        self.subscribers = {}
        self.converters = {}

        self.data = {}
        self.data_times = {}

        self.setup_subscribers()

        self.get_logger().info("cvt node ready")

    def setup_subscribers(self):
        for topic_conf in self.config["topics"]:
            self.data[topic_conf["name"]] = None
            self.data_times[topic_conf["name"]] = -1.0

            self.converters[topic_conf["name"]] = str_to_cvt_class[topic_conf["type"]](
                **topic_conf["args"]
            )
            sub = self.create_subscription(
                self.converters[topic_conf["name"]].msg_type,  # Message type
                topic_conf["topic"],  # Topic name
                lambda msg, topic_conf=topic_conf: self.handle_msg(
                    msg, topic_conf
                ),  # Callback with additional args
                10,  # QoS (default queue size)
            )

    def handle_msg(self, msg, topic_conf):
        self.data[topic_conf["name"]] = msg
        self.data_times[topic_conf["name"]] = get_ns_in_sec()

    def get_data(self, device="cpu"):
        return {k: self.converters[k].cvt(msg) for k, msg in self.data.items()}

    def can_get_data(self):
        return all(
            [
                get_ns_in_sec() - dt < self.config["max_age"]
                for dt in self.data_times.values()
            ]
        )

    def get_status_str(self):
        out = "\n ---converter status--- \n"
        for topic_conf in self.config["topics"]:

            data_exists = self.data[topic_conf["name"]] is not None
            data_age = get_ns_in_sec() - self.data_times[topic_conf["name"]]
            out += "\t{:<16} exists: {} age:{:.2f}s\n".format(
                topic_conf["name"] + " " + topic_conf["topic"] + ":",
                data_exists,
                data_age,
            )

        out += "can get data: {}".format(self.can_get_data())
        out += "\n"
        return out
