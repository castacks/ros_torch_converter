import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import (
    update_info_file, update_timestamp_file,
    read_info_file, read_timestamp_file
)

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp



class MarkerArrayTorch(TorchCoordinatorDataType):
    """
    Torch wrapper for visualization_msgs/MarkerArray.
    Supports both single-pose markers and multi-point markers
    (LINE_STRIP, LINE_LIST).
    """

    to_rosmsg_type = MarkerArray
    from_rosmsg_type = MarkerArray
    time_spec = TimeSpec.SYNC

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Single-point markers
        self.points = torch.zeros(0, 3, device=device)

        # Multi-point markers (list of (Ni x 3) torch tensors)
        self.line_points = []  # None if not a line marker

        # Per-marker attributes
        self.colors = torch.zeros(0, 4, device=device)
        self.scales = torch.zeros(0, 3, device=device)
        self.types = torch.zeros(0, dtype=torch.long, device=device)
        self.ns = []  # list of strings

        self.orientations = None

        # ROS metadata
        self.frame_id = ""
        self.stamp = 0.0

    # ----------------------------------------------------------------------
    #   From Torch
    # ----------------------------------------------------------------------
    @staticmethod
    def from_torch(points,
                   colors=None,
                   scales=None,
                   types=None,
                   ns=None,
                   line_points=None):
        """
        points: (N,3) tensor used for single-position markers
        line_points: list of tensors, each Mx3, for LINE_STRIP markers
        """

        device = points.device
        N = points.shape[0]

        mat = MarkerArrayTorch(device=device)
        mat.points = points

        # Fill attributes
        mat.colors = colors if colors is not None else torch.ones(N, 4, device=device)
        mat.scales = scales if scales is not None else torch.full((N, 3), 0.2, device=device)
        mat.types = types if types is not None else torch.full((N,), Marker.SPHERE, device=device)
        mat.ns = ns if ns is not None else ["marker"] * N

        # Ensure line_points list length matches N
        if line_points is None:
            mat.line_points = [None] * N
        else:
            assert len(line_points) == N
            mat.line_points = line_points

        return mat

    # ----------------------------------------------------------------------
    #   From ROS
    # ----------------------------------------------------------------------
    @staticmethod
    def from_rosmsg(msg, device='cpu'):
        mat = MarkerArrayTorch(device=device)

        for m in msg.markers:
            mat.ns.append(m.ns)
            mat.types = torch.cat([mat.types, torch.tensor([m.type], device=device)])

            # Color
            new_color = torch.tensor(
                [m.color.r, m.color.g, m.color.b, m.color.a],
                device=device
            ).reshape(1, 4)
            mat.colors = torch.cat([mat.colors, new_color], dim=0)

            # Scale
            new_scale = torch.tensor([m.scale.x, m.scale.y, m.scale.z],
                                     device=device).reshape(1, 3)
            mat.scales = torch.cat([mat.scales, new_scale], dim=0)

            # Line markers store in m.points[]
            if m.type in (Marker.LINE_STRIP, Marker.LINE_LIST):
                pts = torch.tensor(
                    [[p.x, p.y, p.z] for p in m.points],
                    device=device
                ).float()
                mat.line_points.append(pts)

                # Dummy single point for compatibility
                mat.points = torch.cat([
                    mat.points,
                    torch.zeros(1, 3, device=device)
                ], dim=0)

            else:
                # Single-pose marker
                new_pt = torch.tensor(
                    [m.pose.position.x,
                     m.pose.position.y,
                     m.pose.position.z],
                    device=device
                ).float().reshape(1, 3)

                mat.points = torch.cat([mat.points, new_pt], dim=0)
                mat.line_points.append(None)

        if msg.markers:
            mat.stamp = stamp_to_time(msg.markers[0].header.stamp)
            mat.frame_id = msg.markers[0].header.frame_id

        return mat

    # ----------------------------------------------------------------------
    #   To ROS
    # ----------------------------------------------------------------------
    def to_rosmsg(self):
        msg = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        clear_marker.header.stamp = time_to_stamp(self.stamp)
        clear_marker.header.frame_id = self.frame_id
        msg.markers.append(clear_marker)

        for i in range(len(self.types)):
            m = Marker()
            m.header.stamp = time_to_stamp(self.stamp)
            m.header.frame_id = self.frame_id
            m.id = i
            m.ns = self.ns[i]
            m.type = int(self.types[i].item())
            m.action = Marker.ADD

            # Colors/scales
            c = self.colors[i]
            m.color.r, m.color.g, m.color.b, m.color.a = c.tolist()

            s = self.scales[i]
            m.scale.x, m.scale.y, m.scale.z = s.tolist()

            # Handle line markers
            if m.type in (Marker.LINE_STRIP, Marker.LINE_LIST):
                pts = self.line_points[i]
                if pts is None:
                    continue  # invalid

                for p in pts:
                    m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))

            else:
                # Single-pose marker
                pt = self.points[i]
                m.pose.position.x = float(pt[0])
                m.pose.position.y = float(pt[1])
                m.pose.position.z = float(pt[2])
                m.pose.orientation.w = 1.0

            if self.orientations[i] is not None:
                q = self.orientations[i]
                m.pose.orientation.x = float(q[0])
                m.pose.orientation.y = float(q[1])
                m.pose.orientation.z = float(q[2])
                m.pose.orientation.w = float(q[3])
            else:
                m.pose.orientation.w = 1.0


            msg.markers.append(m)

        return msg

    # ----------------------------------------------------------------------
    #   KITTI I/O
    # ----------------------------------------------------------------------
    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_info_file(base_dir, idx, 'frame_id', self.frame_id)

        save_fp = os.path.join(base_dir, f"{idx:08d}.txt")

        # Only save single markers, not line points
        arr = torch.cat([self.points, self.colors, self.scales], dim=1)
        np.savetxt(save_fp, arr.cpu().numpy())

    @staticmethod
    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, f"{idx:08d}.txt")
        data = np.loadtxt(fp)

        pts = torch.tensor(data[:, :3], device=device).float()
        cols = torch.tensor(data[:, 3:7], device=device).float()
        scs = torch.tensor(data[:, 7:10], device=device).float()

        mat = MarkerArrayTorch.from_torch(pts, cols, scs)
        mat.stamp = read_timestamp_file(base_dir, idx)
        mat.frame_id = read_info_file(base_dir, idx, 'frame_id')
        return mat

    def rand_init(device='cpu'):
        N = 10  # number of markers

        # Random 3D points
        pts = torch.rand(N, 3, device=device)

        # Random RGB colors (alpha = 1)
        cols = torch.rand(N, 4, device=device)
        cols[:, 3] = 1.0

        # Marker type: 8 = SPHERE
        types = torch.full((N,), 8, device=device)

        # Marker scale (same for all)
        scales = torch.tensor([0.2, 0.2, 0.2], device=device).repeat(N, 1)

        # Build marker array
        mat = MarkerArrayTorch.from_torch(
            points=pts,
            colors=cols,
            types=types,
            scales=scales
        )

        # Metadata
        mat.frame_id = "random_markers"
        mat.stamp = float(np.random.rand())

        return mat

    # ----------------------------------------------------------------------
    #   Misc
    # ----------------------------------------------------------------------
    def __eq__(self, other):
        return (
            self.frame_id == other.frame_id and
            abs(self.stamp - other.stamp) < 1e-8 and
            torch.allclose(self.points, other.points) and
            torch.allclose(self.colors, other.colors) and
            torch.allclose(self.scales, other.scales) and
            torch.allclose(self.types, other.types)
        )

    def to(self, device):
        self.device = device
        self.points = self.points.to(device)
        self.colors = self.colors.to(device)
        self.scales = self.scales.to(device)
        self.types = self.types.to(device)
        self.line_points = [
            None if lp is None else lp.to(device) for lp in self.line_points
        ]
        return self

    def __repr__(self):
        return f"MarkerArrayTorch with {self.points.shape[0]} markers on device {self.device}"
