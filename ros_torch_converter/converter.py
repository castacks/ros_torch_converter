import copy

from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.float import Float32Torch
from ros_torch_converter.datatypes.bool import BoolTorch
from ros_torch_converter.datatypes.image import ImageTorch, FeatureImageTorch, ThermalImageTorch, Thermal16bitImageTorch
from ros_torch_converter.datatypes.intrinsics import IntrinsicsTorch
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch, FeaturePointCloudTorch
from ros_torch_converter.datatypes.transform import TransformTorch, OdomTransformTorch
from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.goal_array import GoalArrayTorch
from ros_torch_converter.datatypes.people_detections import PeopleDetectionsTorch
from ros_torch_converter.datatypes.track_path import TrackPathTorch

from tartandriver_utils.ros_utils import stamp_to_time

str_to_cvt_class = {
    "BEVGrid": BEVGridTorch,
    "Float32": Float32Torch,
    "Bool": BoolTorch,
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
    "GoalArray": GoalArrayTorch,
    "PeopleDetections": PeopleDetectionsTorch,
    "TrackPath": TrackPathTorch,
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
        self.sync_lock = False
        self.synced_topics = set()
        self.syncs = []

        self.setup_subscribers()

        self.get_logger().info("cvt node ready")

    def setup_subscribers(self):
        sync_groups = self.config.get("sync_topics", [])
        if sync_groups:
            self._setup_synchronized_subscribers(sync_groups)

        for topic_conf in self.config["topics"]:
            self.data[topic_conf["name"]] = None
            self.data_times[topic_conf["name"]] = -1.0

            self.converters[topic_conf["name"]] = str_to_cvt_class[topic_conf["type"]]

            if topic_conf["name"] in self.synced_topics:
                continue

            self.create_subscription(
                self.converters[topic_conf["name"]].from_rosmsg_type,  # Message type
                topic_conf["topic"],  # Topic name
                lambda msg, topic_conf=topic_conf: self.handle_msg(
                    msg, topic_conf
                ),  # Callback with additional args
                qos_profile=qos_profile_sensor_data,  # QoS (default queue size)
            )

    def _setup_synchronized_subscribers(self, sync_groups):
        for sync_config in sync_groups:
            topic_names = sync_config["topics"]
            queue_size = sync_config.get("queue_size", 10)
            slop = sync_config.get("slop", 0.05)

            subscribers = []
            topic_configs = []

            for topic_name in topic_names:
                topic_conf = next(
                    (t for t in self.config["topics"] if t["name"] == topic_name),
                    None,
                )
                if topic_conf is None:
                    self.get_logger().warn(
                        f"Sync topic {topic_name} not found in topics list"
                    )
                    continue

                self.data[topic_conf["name"]] = None
                self.data_times[topic_conf["name"]] = -1.0
                self.converters[topic_conf["name"]] = str_to_cvt_class[topic_conf["type"]]

                sub = Subscriber(
                    self,
                    self.converters[topic_name].from_rosmsg_type,
                    topic_conf["topic"],
                    qos_profile=qos_profile_sensor_data,
                )
                subscribers.append(sub)
                topic_configs.append(topic_conf)
                self.synced_topics.add(topic_name)

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
                self.syncs.append(sync)

    def handle_msg(self, msg, topic_conf):
        if not self.lock:
            self.data[topic_conf["name"]] = msg
            try:
                self.data_times[topic_conf["name"]] = stamp_to_time(msg.header.stamp)
            except:
                self.data_times[topic_conf["name"]] = stamp_to_time(self.get_clock().now().to_msg())

    def handle_synchronized_msgs(self, msgs, topic_configs):
        if not self.sync_lock:
            for msg, topic_conf in zip(msgs, topic_configs):
                self.data[topic_conf["name"]] = msg
                try:
                    self.data_times[topic_conf["name"]] = stamp_to_time(msg.header.stamp)
                except:
                    self.data_times[topic_conf["name"]] = stamp_to_time(self.get_clock().now().to_msg())

    def get_data(self, return_times=False, device="cpu"):
        self.lock = True
        self.sync_lock = True
        # Only convert non-None messages (skip optional topics that haven't received data)
        data = {k: self.converters[k].from_rosmsg(msg, device=self.device) 
                for k, msg in self.data.items() if msg is not None}
        times = copy.deepcopy(self.data_times)
        self.lock = False
        self.sync_lock = False

        return (data, times) if return_times else data

    def can_get_data(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())
        
        for topic_conf in self.config["topics"]:
            # Skip optional topics
            if topic_conf.get("optional", False):
                continue
                
            topic_name = topic_conf["name"]

            if self.data[topic_name] is None:
                return False

            data_time = self.data_times[topic_name]
            
            # Check if data is too old
            if curr_time - data_time >= self.config["max_age"]:
                return False
        
        return True

    def get_status_str(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())
        out = "\n ---converter status--- \n"
        for topic_conf in self.config["topics"]:
            data_exists = self.data[topic_conf["name"]] is not None
            data_age = curr_time - self.data_times[topic_conf["name"]]
            is_optional = topic_conf.get("optional", False)
            optional_str = " [OPTIONAL]" if is_optional else ""
            out += "\t{:<16} exists: {} age:{:.2f}s{}\n".format(
                topic_conf["name"] + " " + topic_conf["topic"] + ":",
                data_exists,
                data_age,
                optional_str,
            )

        out += "can get data: {}\n".format(self.can_get_data())
        out += "curr time: {}\n".format(curr_time)
        return out
