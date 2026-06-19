"""Generate lightweight stub packages so ros_torch_converter (and tartandriver_utils)
import and run OFFLINE without a ROS install. These stubs only satisfy module-import-time
references (class attributes like `from_rosmsg_type`, type hints) and the live-node code path
that we never exercise offline. The actual offline conversion deserializes ROS1 bags via the
`rosbags` library (whose message objects are duck-typed), so no real ROS message classes are
needed. Run once: `python offline_stubs/_gen.py`.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Common base: a permissive stub class that accepts any ctor kwargs.
BASE = (
    "class _Stub:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        self.__dict__.update(kwargs)\n"
)

# package_name -> {submodule(None=__init__): [class names]}
PKGS = {
    "sensor_msgs": {"msg": ["Image", "CompressedImage", "CameraInfo", "PointCloud2", "PointField"]},
    "std_msgs": {"msg": ["Float32", "Bool", "Float32MultiArray", "MultiArrayDimension"]},
    "nav_msgs": {"msg": ["Odometry", "Path"]},
    "geometry_msgs": {"msg": ["TwistStamped", "PoseArray", "Pose", "Point", "Quaternion",
                               "PoseStamped", "PoseWithCovarianceStamped", "TransformStamped"]},
    "builtin_interfaces": {"msg": ["Time"]},
    "core_interfaces": {"msg": ["Mission", "Waypoint"]},
    "grid_map_msgs": {"msg": ["GridMap"]},
    "perception_interfaces": {"msg": ["FeatureImage"]},
}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


for pkg, subs in PKGS.items():
    write(os.path.join(HERE, pkg, "__init__.py"), "")
    for sub, classes in subs.items():
        body = BASE + "\n" + "".join(f"class {c}(_Stub):\n    pass\n\n" for c in classes)
        write(os.path.join(HERE, pkg, f"{sub}.py"), body)

# cv_bridge: CvBridge instantiated in image ctors; its convert methods are never called
# on the offline CompressedImage path (that uses cv2.imdecode directly).
write(os.path.join(HERE, "cv_bridge", "__init__.py"),
      "class CvBridge:\n"
      "    def imgmsg_to_cv2(self, *a, **k):\n"
      "        raise NotImplementedError('cv_bridge stub: raw Image path unavailable offline')\n"
      "    def cv2_to_imgmsg(self, *a, **k):\n"
      "        raise NotImplementedError('cv_bridge stub')\n"
      "    def cv2_to_compressed_imgmsg(self, *a, **k):\n"
      "        raise NotImplementedError('cv_bridge stub')\n")

# message_filters: only used by the live ROSTorchConverter node.
write(os.path.join(HERE, "message_filters", "__init__.py"),
      "class Subscriber:\n    def __init__(self, *a, **k):\n        pass\n\n"
      "class ApproximateTimeSynchronizer:\n"
      "    def __init__(self, *a, **k):\n        pass\n"
      "    def registerCallback(self, *a, **k):\n        pass\n")

# rclpy: Node is a base class for the live node; qos/executors only used live.
write(os.path.join(HERE, "rclpy", "__init__.py"),
      "def init(*a, **k):\n    pass\n\ndef shutdown(*a, **k):\n    pass\n\ndef ok(*a, **k):\n    return True\n")
write(os.path.join(HERE, "rclpy", "node.py"),
      "class Node:\n    def __init__(self, *a, **k):\n        pass\n")
write(os.path.join(HERE, "rclpy", "qos.py"),
      "qos_profile_sensor_data = None\n")
write(os.path.join(HERE, "rclpy", "executors.py"),
      "class MultiThreadedExecutor:\n    def __init__(self, *a, **k):\n        pass\n")

print("stubs generated under", HERE)
