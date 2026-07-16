import copy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.float import Float32Torch
from ros_torch_converter.datatypes.bool import BoolTorch
from ros_torch_converter.datatypes.command import CommandTorch
from ros_torch_converter.datatypes.racepak import (
    PedalPosTorch,
    ShockPosTorch,
    WheelRPMTorch,
)
from ros_torch_converter.datatypes.mppi_solution import MPPISolutionTorch
from ros_torch_converter.datatypes.image import (
    ImageTorch,
    CompressedImageTorch,
    FeatureImageTorch,
    ThermalImageTorch,
    Thermal16bitImageTorch,
)
from ros_torch_converter.datatypes.intrinsics import IntrinsicsTorch, CameraInfoTorch
from ros_torch_converter.datatypes.pointcloud import (
    PointCloudTorch,
    FeaturePointCloudTorch,
)
from ros_torch_converter.datatypes.transform import TransformTorch, OdomTransformTorch
from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.goal_array import GoalArrayTorch
from ros_torch_converter.datatypes.path import PathTorch
from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch
from ros_torch_converter.datatypes.people_detections import PeopleDetectionsTorch
from ros_torch_converter.datatypes.sensor_msgs import (
    ImuTorch,
    NavSatFixTorch,
    PoseWithCovarianceTorch,
    TwistTorch,
    FFCStatusTorch,
)
from ros_torch_converter.datatypes.frontier_scores import FrontierScoresTorch

from tartandriver_utils.ros_utils import stamp_to_time

str_to_cvt_class = {
    "BEVGrid": BEVGridTorch,
    "GridMap": BEVGridTorch,  # GridMap is handled by BEVGridTorch
    "Float32": Float32Torch,
    "Bool": BoolTorch,
    "Command": CommandTorch,
    "PedalPos": PedalPosTorch,
    "ShockPos": ShockPosTorch,
    "WheelRPM": WheelRPMTorch,
    "MPPISolution": MPPISolutionTorch,
    "Image": ImageTorch,
    "CompressedImage": CompressedImageTorch,
    "FeatureImage": FeatureImageTorch,
    "ThermalImage": ThermalImageTorch,
    "Thermal16bitImage": Thermal16bitImageTorch,
    "Intrinsics": IntrinsicsTorch,
    "CameraInfo": CameraInfoTorch,
    "PointCloud": PointCloudTorch,
    "FeaturePointCloud": FeaturePointCloudTorch,
    "Transform": TransformTorch,
    "OdomTransform": OdomTransformTorch,
    "OdomRBState": OdomRBStateTorch,
    "GoalArray": GoalArrayTorch,
    "Path": PathTorch,
    "VoxelGrid": VoxelGridTorch,
    "Imu": ImuTorch,
    "NavSatFix": NavSatFixTorch,
    "PoseWithCovarianceStamped": PoseWithCovarianceTorch,
    "TwistStamped": TwistTorch,
    "FFCStatus": FFCStatusTorch,
    "PeopleDetections": PeopleDetectionsTorch,
    "FrontierScores": FrontierScoresTorch,
}


class ROSTorchConverter(Node):
    """Top-level class that manages conversion from ROS->torch.

    Essentially, this class will spin up a number of subscribers and store the latest message for each.
    When it is asked for data, it will convert all the messages to torch and return them as a (potentially nested) dict
    """

    def __init__(self, config, name=""):
        super().__init__(name + "_ros_torch_converter_node")

        self.config = config
        self.device = self.config["device"]
        self.subscribers = {}
        self.sync_filters = []
        self.converters = {}

        self.data = {}
        self.data_times = {}

        self.lock = False
        self.sync_lock = False
        self.synced_topics = set()

        self.setup_subscribers()

        self.get_logger().info("cvt node ready")

    def _topic_key(self, topic_conf):
        if topic_conf.get("group"):
            return f"{topic_conf['group']}/{topic_conf['name']}"
        return topic_conf["name"]

    def _topic_config_by_key_or_name(self, topic_name):
        for topic_conf in self.config["topics"]:
            if topic_conf["name"] == topic_name or self._topic_key(topic_conf) == topic_name:
                return topic_conf
        return None

    def setup_subscribers(self):
        sync_groups = self.config.get("sync_topics", [])

        for topic_conf in self.config["topics"]:
            tname = self._topic_key(topic_conf)
            self.data[tname] = None
            self.data_times[tname] = -1.0
            self.converters[tname] = str_to_cvt_class[topic_conf["type"]]

        if sync_groups:
            self._setup_synchronized_subscribers(sync_groups)

        for topic_conf in self.config["topics"]:
            tname = self._topic_key(topic_conf)
            if tname not in self.synced_topics:
                sub = self.create_subscription(
                    self.converters[tname].from_rosmsg_type,
                    topic_conf["topic"],
                    lambda msg, topic_conf=topic_conf: self.handle_msg(msg, topic_conf),
                    qos_profile=qos_profile_sensor_data,
                )
                self.subscribers[tname] = sub

    def _setup_synchronized_subscribers(self, sync_groups):
        for sync_config in sync_groups:
            topic_names = sync_config["topics"]
            queue_size = sync_config.get("queue_size", 5)
            slop = sync_config.get("slop", 0.1)

            subscribers = []
            topic_configs = []

            for topic_name in topic_names:
                topic_conf = self._topic_config_by_key_or_name(topic_name)
                if topic_conf is None:
                    self.get_logger().warn(
                        f"Sync topic {topic_name} not found in topics list"
                    )
                    continue

                tname = self._topic_key(topic_conf)
                sub = Subscriber(
                    self,
                    self.converters[tname].from_rosmsg_type,
                    topic_conf["topic"],
                )
                subscribers.append(sub)
                topic_configs.append(topic_conf)
                self.subscribers[tname] = sub
                self.synced_topics.add(tname)

            if len(subscribers) > 1:
                sync = ApproximateTimeSynchronizer(
                    subscribers,
                    queue_size=queue_size,
                    slop=slop,
                )
                sync.registerCallback(
                    lambda *msgs, configs=topic_configs: self.handle_synchronized_msgs(
                        msgs, configs
                    )
                )
                self.sync_filters.append(sync)

    def handle_msg(self, msg, topic_conf):
        tname = self._topic_key(topic_conf)
        if not self.lock:
            self.data[tname] = msg
            try:
                self.data_times[tname] = stamp_to_time(msg.header.stamp)
            except:
                self.data_times[tname] = stamp_to_time(self.get_clock().now().to_msg())

    def handle_synchronized_msgs(self, msgs, topic_configs):
        if not self.sync_lock:
            for msg, topic_conf in zip(msgs, topic_configs):
                tname = self._topic_key(topic_conf)
                self.data[tname] = msg
                try:
                    self.data_times[tname] = stamp_to_time(msg.header.stamp)
                except:
                    self.data_times[tname] = stamp_to_time(
                        self.get_clock().now().to_msg()
                    )

    def get_data(self, return_times=False, device="cpu"):
        self.lock = True
        self.sync_lock = True
        data = {}

        for topic_conf in self.config["topics"]:
            tname = self._topic_key(topic_conf)
            msg = self.data[tname]
            if msg is None:
                continue
            cvt = self.converters[tname]
            args = {
                k: v
                for k, v in topic_conf.get("args", {}).items()
                if k != "interpolation"
            }
            data[tname] = cvt.from_rosmsg(msg, device=self.device, **args)

        times = copy.deepcopy(self.data_times)
        self.lock = False
        self.sync_lock = False

        return (data, times) if return_times else data

    def can_get_data(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())

        for topic_conf in self.config["topics"]:
            if topic_conf.get("optional", False):
                continue

            tname = self._topic_key(topic_conf)
            if self.data[tname] is None:
                return False

            max_age = topic_conf.get("max_age", self.config.get("max_age"))
            data_time = self.data_times[tname]

            if data_time < 0.0:
                return False
            if max_age is not None and curr_time - data_time >= max_age:
                return False

        return True

    def get_status_str(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())
        out = "\n ---converter status--- \n"
        for topic_conf in self.config["topics"]:
            tname = self._topic_key(topic_conf)
            data_exists = self.data[tname] is not None
            data_age = curr_time - self.data_times[tname]
            is_optional = topic_conf.get("optional", False)
            optional_str = " [OPTIONAL]" if is_optional else ""
            out += "\t{:<16} exists: {} age:{:.2f}s{}\n".format(
                tname + " " + topic_conf["topic"] + ":",
                data_exists,
                data_age,
                optional_str,
            )

        out += "can get data: {}\n".format(self.can_get_data())
        out += "curr time: {}\n".format(curr_time)
        return out
