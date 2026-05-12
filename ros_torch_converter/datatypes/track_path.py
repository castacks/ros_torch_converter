"""TrackPathTorch — a dense reference path consumed by the path-tracking
MPPI cost terms (PathTrackError, PathProgress, PathYawAlignment).

Wraps `nav_msgs/Path`. Stored as a single `[P, 7]` tensor of
(x, y, z, qx, qy, qz, qw) — full quaternion, no information loss at the
ros↔torch boundary.

Layout matches the canonical `PathTorch` upstream wrapper for nav_msgs/Path.

Path-empty messages (planner FAILED → empty `nav_msgs/Path`, see
voxel_rrt_planner_node's failure branch) decode to a `[0, 7]` tensor,
which the cost terms treat as "infeasible-and-bail" rather than crashing
on `torch.stack` of an empty list.
"""

import os
import numpy as np
import torch

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp


# Column count of the stored tensor. Centralised so consumers (cost terms,
# tests) can sanity-check shapes without magic numbers.
POSE_DIM = 7  # x, y, z, qx, qy, qz, qw


class TrackPathTorch(TorchCoordinatorDataType):
    """A dense reference path (xyz + full quat, [P, 7])."""

    to_rosmsg_type = Path
    from_rosmsg_type = Path

    def __init__(self, device="cpu"):
        super().__init__()
        self.poses = torch.zeros(0, POSE_DIM, device=device)
        self.device = device

    @staticmethod
    def from_torch(poses):
        """Build a TrackPathTorch directly from a [P, 7] tensor."""
        if poses.ndim != 2 or poses.shape[1] != POSE_DIM:
            raise ValueError(
                "TrackPathTorch.from_torch: expected [P, {}], got {}".format(
                    POSE_DIM, tuple(poses.shape)
                )
            )
        pat = TrackPathTorch(device=poses.device)
        pat.poses = poses
        return pat

    @staticmethod
    def from_rosmsg(msg, device="cpu"):
        res = TrackPathTorch(device)

        if len(msg.poses) > 0:
            arr = np.empty((len(msg.poses), POSE_DIM), dtype=np.float32)
            for i, ps in enumerate(msg.poses):
                p = ps.pose.position
                q = ps.pose.orientation
                arr[i, 0] = p.x
                arr[i, 1] = p.y
                arr[i, 2] = p.z
                arr[i, 3] = q.x
                arr[i, 4] = q.y
                arr[i, 5] = q.z
                arr[i, 6] = q.w
            res.poses = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        return res

    def to_rosmsg(self):
        msg = Path()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        for row in self.poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(row[0].item())
            ps.pose.position.y = float(row[1].item())
            ps.pose.position.z = float(row[2].item())
            ps.pose.orientation = Quaternion(
                x=float(row[3].item()),
                y=float(row[4].item()),
                z=float(row[5].item()),
                w=float(row[6].item()),
            )
            msg.poses.append(ps)
        return msg

    def to_kitti(self, base_dir, idx):
        """Persist as a single .txt of the [P, 7] tensor."""
        save_fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        np.savetxt(save_fp, self.poses.cpu().numpy())

    @staticmethod
    def from_kitti(base_dir, idx, device="cpu"):
        fp = os.path.join(base_dir, "{:08d}.txt".format(idx))
        data = np.loadtxt(fp)
        # np.loadtxt collapses a single-row file to 1D; restore [1, 7].
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Empty file (P=0) -> np.loadtxt returns shape (0,); reshape to
        # the canonical [0, 7] so downstream shape checks still pass.
        if data.size == 0:
            data = np.zeros((0, POSE_DIM), dtype=np.float32)
        tensor = torch.tensor(data, dtype=torch.float32, device=device)
        return TrackPathTorch.from_torch(tensor)

    @staticmethod
    def rand_init(device="cpu"):
        """Random fixture for tests / smoke-checks.

        Generates a 10-pose path with normalized random quaternions so
        consumers that re-derive yaw from the quat columns get a
        meaningful (and bounded) value rather than garbage.
        """
        xyz = torch.rand(10, 3, device=device)
        quat = torch.randn(10, 4, device=device)
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-9)
        poses = torch.cat([xyz, quat], dim=-1)
        pat = TrackPathTorch.from_torch(poses)
        pat.frame_id = "random"
        pat.stamp = float(np.random.rand())
        return pat

    def __eq__(self, other):
        if not isinstance(other, TrackPathTorch):
            return NotImplemented
        if self.frame_id != other.frame_id:
            return False
        if abs(self.stamp - other.stamp) > 1e-8:
            return False
        if self.poses.shape != other.poses.shape:
            return False
        return torch.allclose(self.poses, other.poses)

    def to(self, device):
        self.device = device
        self.poses = self.poses.to(device)
        return self

    def __repr__(self):
        return "TrackPathTorch(shape={}, device={}, frame={})".format(
            tuple(self.poses.shape), self.device, self.frame_id
        )
