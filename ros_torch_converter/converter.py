import copy
import warnings
import importlib

# Live-node-only deps (rclpy/message_filters). Needed when running the ROS2 ROSTorchConverter
# node, but NOT for offline ROS1-bag -> KITTI conversion. Guard them so the offline path
# (which only needs `str_to_cvt_class`) imports without a ROS install.
try:
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from message_filters import ApproximateTimeSynchronizer, Subscriber
except Exception:  # pragma: no cover - offline path
    Node = object
    qos_profile_sensor_data = None
    ApproximateTimeSynchronizer = Subscriber = None

try:
    from tartandriver_utils.ros_utils import stamp_to_time
except Exception:  # pragma: no cover
    stamp_to_time = None


def _opt_import(modpath, *names):
    """Import `names` from `modpath`, returning {name: obj}. If the module can't be imported
    (e.g. its exotic ROS/C++ deps like ros2_numpy_cpp are absent in an offline, ROS-free env),
    warn and return {} so the datatypes that DO import remain usable. This lets ROS1->KITTI
    conversion run with only the converters its config actually references."""
    try:
        mod = importlib.import_module(modpath)
        return {n: getattr(mod, n) for n in names}
    except Exception as e:  # pragma: no cover - offline path
        warnings.warn("ros_torch_converter: datatype module '{}' unavailable ({}: {}); "
                      "its converters will be skipped.".format(modpath, type(e).__name__, e))
        return {}


# name(s) registered in str_to_cvt_class -> (module path, class name)
_REGISTRY_SPEC = {
    "BEVGrid": ("ros_torch_converter.datatypes.bev_grid", "BEVGridTorch"),
    "GridMap": ("ros_torch_converter.datatypes.bev_grid", "BEVGridTorch"),
    "Float32": ("ros_torch_converter.datatypes.float", "Float32Torch"),
    "Bool": ("ros_torch_converter.datatypes.bool", "BoolTorch"),
    "Command": ("ros_torch_converter.datatypes.command", "CommandTorch"),
    "PedalPos": ("ros_torch_converter.datatypes.racepak", "PedalPosTorch"),
    "ShockPos": ("ros_torch_converter.datatypes.racepak", "ShockPosTorch"),
    "WheelRPM": ("ros_torch_converter.datatypes.racepak", "WheelRPMTorch"),
    "MPPISolution": ("ros_torch_converter.datatypes.mppi_solution", "MPPISolutionTorch"),
    "Image": ("ros_torch_converter.datatypes.image", "ImageTorch"),
    "CompressedImage": ("ros_torch_converter.datatypes.image", "CompressedImageTorch"),
    "FeatureImage": ("ros_torch_converter.datatypes.image", "FeatureImageTorch"),
    "ThermalImage": ("ros_torch_converter.datatypes.image", "ThermalImageTorch"),
    "Thermal16bitImage": ("ros_torch_converter.datatypes.image", "Thermal16bitImageTorch"),
    "Thermal8bitImage": ("ros_torch_converter.datatypes.image", "Thermal8bitImageTorch"),
    "Thermal16bitCompressedImage": ("ros_torch_converter.datatypes.image", "CompressedThermal16bitImageTorch"),
    "DepthImage": ("ros_torch_converter.datatypes.depth", "DepthImageTorch"),
    "Intrinsics": ("ros_torch_converter.datatypes.intrinsics", "IntrinsicsTorch"),
    "CameraInfo": ("ros_torch_converter.datatypes.intrinsics", "CameraInfoTorch"),
    "PointCloud": ("ros_torch_converter.datatypes.pointcloud", "PointCloudTorch"),
    "FeaturePointCloud": ("ros_torch_converter.datatypes.pointcloud", "FeaturePointCloudTorch"),
    "Transform": ("ros_torch_converter.datatypes.transform", "TransformTorch"),
    "OdomTransform": ("ros_torch_converter.datatypes.transform", "OdomTransformTorch"),
    "OdomRBState": ("ros_torch_converter.datatypes.rb_state", "OdomRBStateTorch"),
    "GoalArray": ("ros_torch_converter.datatypes.goal_array", "GoalArrayTorch"),
    "Path": ("ros_torch_converter.datatypes.path", "PathTorch"),
    "VoxelGrid": ("ros_torch_converter.datatypes.voxel_grid", "VoxelGridTorch"),
    "Imu": ("ros_torch_converter.datatypes.sensor_msgs", "ImuTorch"),
    "NavSatFix": ("ros_torch_converter.datatypes.sensor_msgs", "NavSatFixTorch"),
    "PoseWithCovarianceStamped": ("ros_torch_converter.datatypes.sensor_msgs", "PoseWithCovarianceTorch"),
    "TwistStamped": ("ros_torch_converter.datatypes.sensor_msgs", "TwistTorch"),
    "FFCStatus": ("ros_torch_converter.datatypes.sensor_msgs", "FFCStatusTorch"),
}

# Cache per-module imports so each module is only attempted once.
_module_cache = {}
str_to_cvt_class = {}
for _key, (_mod, _cls) in _REGISTRY_SPEC.items():
    if _mod not in _module_cache:
        _module_cache[_mod] = _opt_import(_mod, *[
            c for k, (m, c) in _REGISTRY_SPEC.items() if m == _mod
        ])
    if _cls in _module_cache[_mod]:
        str_to_cvt_class[_key] = _module_cache[_mod][_cls]


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
        self.converters = {}

        self.data = {}
        self.data_times = {}

        self.lock = False
        self.sync_lock = False
        self.synced_topics = set()

        self.setup_subscribers()

        self.get_logger().info("cvt node ready")

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
                sub = self.create_subscription(
                    self.converters[tname].from_rosmsg_type, # Message type
                    topic_conf["topic"], # Topic name
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
                topic_conf = next((t for t in self.config["topics"] if t["name"] == topic_name), None)
                if topic_conf is None:
                    self.get_logger().warn(f"Sync topic {topic_name} not found in topics list")
                    continue
                
                sub = Subscriber(
                    self,
                    self.converters[topic_name].from_rosmsg_type,
                    topic_conf["topic"]
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
                self.data_times[tname] = stamp_to_time(self.get_clock().now().to_msg())

    def handle_synchronized_msgs(self, msgs, topic_configs):
        tname = f"{topic_conf['group']}/{topic_conf['name']}"
        if not self.sync_lock:
            for msg, topic_conf in zip(msgs, topic_configs):
                self.data[tname] = msg
                try:
                    self.data_times[tname] = stamp_to_time(msg.header.stamp)
                except:
                    self.data_times[tname] = stamp_to_time(self.get_clock().now().to_msg())

    def get_data(self, return_times=False, device="cpu"):
        self.lock = True
        self.sync_lock = True
        data = {}

        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            
            cvt = self.converters[tname]
            msg = self.data[tname]
            #TODO probably need to actually support interpolation here
            args = {k:v for k,v in topic_conf['args'].items() if k != 'interpolation'}
            msg_torch = cvt.from_rosmsg(msg, device=self.device, **args)
            data[tname] = msg_torch

        # data = {k: self.converters[k].from_rosmsg(msg, device=self.device, **self.config["topics"][k]["args"]) for k, msg in self.data.items()}
        times = copy.deepcopy(self.data_times)
        self.lock = False
        self.sync_lock = False

        return (data, times) if return_times else data

    def can_get_data(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())

        for topic_conf in self.config["topics"]:
            tname = f"{topic_conf['group']}/{topic_conf['name']}"
            max_age = topic_conf["max_age"]
            data_time = self.data_times[tname]

            if curr_time - data_time > max_age or data_time < 0.0:
                return False

        return True

    def get_status_str(self):
        curr_time = stamp_to_time(self.get_clock().now().to_msg())
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
