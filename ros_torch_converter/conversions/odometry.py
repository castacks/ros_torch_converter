import torch

from nav_msgs.msg import Odometry

from ros_torch_converter.conversions.base import Conversion


class OdometryToPoseTwist(Conversion):
    """Convert Odometry to 13d pose and twist

    Stacked as [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
    """

    def __init__(self):
        pass

    @property
    def msg_type(self):
        return Odometry

    def cvt(self, msg):
        return torch.tensor(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ]
        )


class OdometryToPose(Conversion):
    """Convert Odometry to 7d pose

    Stacked as [x,y,z, qx,qy,qz,qw]
    """

    def __init__(self):
        pass

    @property
    def msg_type(self):
        return Odometry

    def cvt(self, msg):
        return torch.tensor(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
