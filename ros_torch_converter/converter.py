import copy
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.bev_grid_uint8 import BEVGridUInt8Torch
from ros_torch_converter.datatypes.float import Float32Torch
from ros_torch_converter.datatypes.command import CommandTorch
from ros_torch_converter.datatypes.image import (
    ImageTorch,
    CompressedImageTorch,
    # FeatureImageTorch,
    # ThermalImageTorch,
    # Thermal16bitImageTorch,
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
# from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch
from ros_torch_converter.datatypes.sensor_msgs import (
    ImuTorch,
    NavSatFixTorch,
    PoseWithCovarianceTorch,
    TwistTorch,
    FFCStatusTorch,
)

from tartandriver_utils.ros_utils import stamp_to_time

str_to_cvt_class = {
    "BEVGrid": BEVGridTorch,
    "BEVGridUInt8": BEVGridUInt8Torch,
    "GridMap": BEVGridTorch,
    "Float32": Float32Torch,
    "Command": CommandTorch,
    "Image": ImageTorch,
    "CompressedImage": CompressedImageTorch,
    # "FeatureImage": FeatureImageTorch,
    # "ThermalImage": ThermalImageTorch,
    # "Thermal16bitImage": Thermal16bitImageTorch,
    "Intrinsics": IntrinsicsTorch,
    "CameraInfo": CameraInfoTorch,
    "PointCloud": PointCloudTorch,
    "FeaturePointCloud": FeaturePointCloudTorch,
    "Transform": TransformTorch,
    "OdomTransform": OdomTransformTorch,
    "OdomRBState": OdomRBStateTorch,
    "GoalArray": GoalArrayTorch,
    "Path": PathTorch,
    # "VoxelGrid": VoxelGridTorch,
    "Imu": ImuTorch,
    "NavSatFix": NavSatFixTorch,
    "PoseWithCovarianceStamped": PoseWithCovarianceTorch,
    "TwistStamped": TwistTorch,
    "FFCStatus": FFCStatusTorch,
}


class ROSTorchConverter(object):
    """Top-level class that manages conversion from ROS1->torch."""

    def __init__(self, config, name=""):
        # In ROS1, we initialize the node globally if it hasn't been already
        try:
            rospy.init_node(name + "_ros_torch_converter_node", anonymous=True)
        except rospy.exceptions.ROSException:
            pass # Already initialized elsewhere

        self.config = config
        self.device = self.config["device"]
        self.subscribers = {}
        self.converters = {}

        self.data = {}
        self.data_times = {}

        self.lock = False
        self.sync_lock = False
        self.synced_topics = set()

        self.setup_subscribers()

        rospy.loginfo("cvt node ready")

    def setup_subscribers(self):
        sync_groups = self.config.get("sync_topics", [])
        
        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            self.data[tname] = None
            self.data_times[tname] = -1.0
            self.converters[tname] = str_to_cvt_class[topic_conf["type"]]

        if sync_groups:
            self._setup_synchronized_subscribers(sync_groups)
        
        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            if tname not in self.synced_topics:
                # ROS1 rospy.Subscriber syntax (no qos_profile, uses buff_size/queue_size if needed)
                sub = rospy.Subscriber(
                    topic_conf["topic"], 
                    self.converters[tname].from_rosmsg_type, 
                    lambda msg, topic_conf=topic_conf: self.handle_msg(msg, topic_conf),
                    queue_size=1
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
                topic_conf = next((t for t in self.config["topics"] if t["name"] == topic_name), None)
                if topic_conf is None:
                    rospy.logwarn(f"Sync topic {topic_name} not found in topics list")
                    continue
                
                # ROS1 message_filters Subscriber does not take the 'self' node instance
                sub = Subscriber(
                    topic_conf["topic"],
                    self.converters[topic_name].from_rosmsg_type
                )
                subscribers.append(sub)
                topic_configs.append(topic_conf)
                self.synced_topics.add(topic_name)
            
            if len(subscribers) > 1:
                sync = ApproximateTimeSynchronizer(
                    subscribers,
                    queue_size=queue_size,
                    slop=slop
                )
                sync.registerCallback(lambda *msgs, configs=topic_configs: self.handle_synchronized_msgs(msgs, configs))

    def handle_msg(self, msg, topic_conf):
        tname = f"{topic_conf['group']}/{topic_conf['name']}"
        if not self.lock:
            self.data[tname] = msg
            try:
                self.data_times[tname] = stamp_to_time(msg.header.stamp)
            except:
                self.data_times[tname] = stamp_to_time(rospy.get_rostime())

    def handle_synchronized_msgs(self, msgs, topic_configs):
        if not self.sync_lock:
            for msg, topic_conf in zip(msgs, topic_configs):
                tname = f"{topic_conf['group']}/{topic_conf['name']}"
                self.data[tname] = msg
                try:
                    self.data_times[tname] = stamp_to_time(msg.header.stamp)
                except:
                    self.data_times[tname] = stamp_to_time(rospy.get_rostime())

    def get_data(self, return_times=False, device="cpu"):
        self.lock = True
        self.sync_lock = True
        data = {}

        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            
            cvt = self.converters[tname]
            msg = self.data[tname]
            args = {k:v for k,v in topic_conf['args'].items() if k != 'interpolation'}
            msg_torch = cvt.from_rosmsg(msg, device=self.device, **args)
            data[tname] = msg_torch

        times = copy.deepcopy(self.data_times)
        self.lock = False
        self.sync_lock = False

        return (data, times) if return_times else data

    def can_get_data(self):
        curr_time = stamp_to_time(rospy.get_rostime())

        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            max_age = topic_conf["max_age"]
            data_time = self.data_times[tname]

            if curr_time - data_time > max_age or data_time < 0.0:
                return False

        return True

    def get_status_str(self):
        curr_time = stamp_to_time(rospy.get_rostime())
        out = "\n ---converter status--- \n"
        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            data_exists = self.data[tname] is not None
            data_age = curr_time - self.data_times[tname]
            out += "\t{:<16} exists: {} age:{:.2f}s\n".format(
                tname + " " + topic_conf["topic"] + ":",
                data_exists,
                data_age,
            )

        out += "can get data: {}\n".format(self.can_get_data())
        out += "curr time: {}\n".format(curr_time)
        return out