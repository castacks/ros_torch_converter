"""VoxelCostFieldTorch: sparse per-voxel cost + flags + surface_type.

ROS message: perception_interfaces/VoxelCostGrid.
Sent from the TC-side VoxelCostGridNode to the standalone voxel_rrt_planner.
"""

from __future__ import annotations

import array

import numpy as np
import torch

from ros_torch_converter.datatypes.base import TimeSpec, TorchCoordinatorDataType

from geometry_msgs.msg import Vector3
from perception_interfaces.msg import VoxelCostGrid, VoxelGridMetadata

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp


# Flag bits must match perception_interfaces/msg/VoxelCostGrid.msg
FLAG_DRIVABLE = 1
FLAG_Z_KILLED = 2
FLAG_XY_KILLED = 4
FLAG_RISER_FILLED = 8


class VoxelCostFieldTorch(TorchCoordinatorDataType):
    """Sparse per-voxel cost + flags + SurfaceType produced by VoxelCostGridNode."""

    to_rosmsg_type = VoxelCostGrid
    from_rosmsg_type = VoxelCostGrid
    time_spec = TimeSpec.SYNC

    def __init__(self, device):
        super().__init__()
        self.device = device
        # VoxelGrid metadata (origin/length/resolution as torch tensors).
        self.origin = torch.zeros(3, device=device)
        self.length = torch.zeros(3, device=device)
        self.resolution = torch.zeros(3, device=device)
        self.N = torch.zeros(3, dtype=torch.long, device=device)
        # Sparse arrays aligned with voxel_grid.raster_indices (length M).
        self.grid_indices = torch.zeros(0, 3, dtype=torch.long, device=device)
        self.cost = torch.zeros(0, dtype=torch.float32, device=device)
        self.flags = torch.zeros(0, dtype=torch.uint8, device=device)
        self.surface_type = torch.zeros(0, dtype=torch.int32, device=device)
        # Inclination in degrees; carried for viz. 0 at voxels we never classified.
        self.inclination_deg = torch.zeros(0, dtype=torch.float32, device=device)
        # Roughness (meters): std of signed distance to best-fit PCA plane of
        # the local kernel_size^3 neighborhood. 0 at voxels we never classified.
        self.roughness = torch.zeros(0, dtype=torch.float32, device=device)

    # ---------------- construction ----------------

    @classmethod
    def from_cost_grid_result(cls, voxel_grid, drivable_result, cost_result, device):
        """Build from outputs of `voxel_3d.cost_grid.create_cost_grid`."""
        res = cls(device=device)
        md = voxel_grid.metadata
        res.origin = md.origin.detach().to(device)
        res.length = md.length.detach().to(device) if hasattr(md, "length") else torch.zeros(3, device=device)
        res.resolution = md.resolution.detach().to(device)
        res.N = md.N.detach().to(device)

        gi = voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices).detach().to(device)
        res.grid_indices = gi.long()

        n_vox = gi.shape[0]
        res.cost = cost_result.voxel_cost.to(device=device, dtype=torch.float32)

        flags = torch.zeros(n_vox, dtype=torch.uint8, device=device)
        flags[drivable_result.drivable_mask] |= FLAG_DRIVABLE
        flags[cost_result.z_killed_voxel] |= FLAG_Z_KILLED
        flags[cost_result.xy_killed_voxel] |= FLAG_XY_KILLED
        flags[cost_result.riser_filled_voxel] |= FLAG_RISER_FILLED
        res.flags = flags

        res.surface_type = drivable_result.surface_type_labels.to(device=device, dtype=torch.int32)
        
        incl = getattr(drivable_result, "inclination_deg", None)
        if incl is None or incl.numel() != n_vox:
            incl = torch.zeros(n_vox, dtype=torch.float32, device=device)
        else:
            incl = incl.to(device=device, dtype=torch.float32)
        res.inclination_deg = incl
        
        rough_src = getattr(drivable_result, "roughness", None)
        if rough_src is None or rough_src.numel() != n_vox:
            rough = torch.zeros(n_vox, dtype=torch.float32, device=device)
        else:
            rough = rough_src.to(device=device, dtype=torch.float32)
        res.roughness = rough
        
        return res

    # ---------------- ROS round-trip ----------------

    def to_rosmsg(self):
        msg = VoxelCostGrid()
        msg.header.stamp = time_to_stamp(self.stamp)
        msg.header.frame_id = self.frame_id

        msg.metadata = VoxelGridMetadata()
        msg.metadata.origin = Vector3(
            x=float(self.origin[0].item()),
            y=float(self.origin[1].item()),
            z=float(self.origin[2].item()),
        )
        msg.metadata.length = Vector3(
            x=float(self.length[0].item()),
            y=float(self.length[1].item()),
            z=float(self.length[2].item()),
        )
        msg.metadata.resolution = Vector3(
            x=float(self.resolution[0].item()),
            y=float(self.resolution[1].item()),
            z=float(self.resolution[2].item()),
        )

        gi = np.ascontiguousarray(
            self.grid_indices.detach().cpu().numpy().astype(np.int32)
        ).reshape(-1)
        cost = np.ascontiguousarray(
            self.cost.detach().cpu().numpy().astype(np.float32)
        )
        flags = np.ascontiguousarray(
            self.flags.detach().cpu().numpy().astype(np.uint8)
        )
        stype = np.ascontiguousarray(
            self.surface_type.detach().cpu().numpy().astype(np.int32)
        )
        incl = np.ascontiguousarray(
            self.inclination_deg.detach().cpu().numpy().astype(np.float32)
        )
        rough = np.ascontiguousarray(
            self.roughness.detach().cpu().numpy().astype(np.float32)
        )

        msg.num_voxels = int(cost.shape[0])
        # rclpy requires array.array for typed []: use frombytes() for a C-speed
        # memcpy instead of array.array(typecode, ndarray.tolist()) which is an
        # O(N) Python loop (~200ms for 50k voxels).
        gi_a = array.array("i")
        gi_a.frombytes(gi.tobytes())
        msg.grid_indices_flat = gi_a
        cost_a = array.array("f")
        cost_a.frombytes(cost.tobytes())
        msg.cost = cost_a
        msg.flags = flags.tobytes()
        stype_a = array.array("i")
        stype_a.frombytes(stype.tobytes())
        msg.surface_type = stype_a
        incl_a = array.array("f")
        incl_a.frombytes(incl.tobytes())
        msg.inclination_deg = incl_a
        rough_a = array.array("f")
        rough_a.frombytes(rough.tobytes())
        msg.roughness = rough_a
        return msg

    @staticmethod
    def from_rosmsg(msg, device="cpu", feature_keys=None):
        # Tolerate positional-call pattern used by ROSTorchConverter.
        if isinstance(device, dict) or (isinstance(device, str) and feature_keys is None):
            pass
        if not isinstance(msg, VoxelCostGrid):
            return None
        res = VoxelCostFieldTorch(device=device)
        res.origin = torch.tensor(
            [msg.metadata.origin.x, msg.metadata.origin.y, msg.metadata.origin.z],
            dtype=torch.float32, device=device,
        )
        res.length = torch.tensor(
            [msg.metadata.length.x, msg.metadata.length.y, msg.metadata.length.z],
            dtype=torch.float32, device=device,
        )
        res.resolution = torch.tensor(
            [msg.metadata.resolution.x, msg.metadata.resolution.y, msg.metadata.resolution.z],
            dtype=torch.float32, device=device,
        )
        # Infer N from length / resolution.
        res.N = torch.round(res.length / torch.clamp(res.resolution, min=1e-6)).long()

        n = int(msg.num_voxels)
        if n == 0:
            res.grid_indices = torch.zeros(0, 3, dtype=torch.long, device=device)
            res.cost = torch.zeros(0, dtype=torch.float32, device=device)
            res.flags = torch.zeros(0, dtype=torch.uint8, device=device)
            res.surface_type = torch.zeros(0, dtype=torch.int32, device=device)
            res.inclination_deg = torch.zeros(0, dtype=torch.float32, device=device)
            res.roughness = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            gi = np.frombuffer(bytes(msg.grid_indices_flat), dtype=np.int32).reshape(-1, 3)
            res.grid_indices = torch.from_numpy(gi.astype(np.int64)).to(device)
            res.cost = torch.from_numpy(
                np.frombuffer(bytes(msg.cost), dtype=np.float32).astype(np.float32)
            ).to(device)
            res.flags = torch.from_numpy(
                np.frombuffer(bytes(msg.flags), dtype=np.uint8).astype(np.uint8)
            ).to(device)
            res.surface_type = torch.from_numpy(
                np.frombuffer(bytes(msg.surface_type), dtype=np.int32).astype(np.int32)
            ).to(device)
            incl_src = getattr(msg, "inclination_deg", None)
            if incl_src is None or len(incl_src) == 0:
                res.inclination_deg = torch.zeros(n, dtype=torch.float32, device=device)
            else:
                res.inclination_deg = torch.from_numpy(
                    np.frombuffer(bytes(incl_src), dtype=np.float32).astype(np.float32)
                ).to(device)
            rough_src = getattr(msg, "roughness", None)
            if rough_src is None or len(rough_src) == 0:
                res.roughness = torch.zeros(n, dtype=torch.float32, device=device)
            else:
                res.roughness = torch.from_numpy(
                    np.frombuffer(bytes(rough_src), dtype=np.float32).astype(np.float32)
                ).to(device)
        res.frame_id = msg.header.frame_id
        res.stamp = stamp_to_time(msg.header.stamp)
        return res

    # ---------------- utilities ----------------

    def to_dense_cost(self) -> torch.Tensor:
        """Scatter the sparse cost into a dense (Nx, Ny, Nz) float32 tensor.

        Returns 0-initialized float32 tensor where non-occupied voxels are 0
        (treated as impassable by the RRT when use_costs=True).
        """
        Nx, Ny, Nz = int(self.N[0].item()), int(self.N[1].item()), int(self.N[2].item())
        dense = torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device=self.device)
        if self.cost.numel() == 0:
            return dense
        gi = self.grid_indices
        dense[gi[:, 0], gi[:, 1], gi[:, 2]] = self.cost
        return dense

    # ---------------- kitti / device ----------------

    def to_kitti(self, base_dir, idx):
        pass

    def from_kitti(self, base_dir, idx, device):
        pass

    @staticmethod
    def rand_init(device="cpu"):
        res = VoxelCostFieldTorch(device=device)
        res.origin = torch.tensor([-5.0, -5.0, -2.0], device=device)
        res.length = torch.tensor([10.0, 10.0, 4.0], device=device)
        res.resolution = torch.tensor([0.2, 0.2, 0.2], device=device)
        res.N = torch.round(res.length / res.resolution).long()
        res.grid_indices = torch.randint(
            low=0, high=20, size=(100, 3), device=device
        )
        res.cost = torch.rand(100, dtype=torch.float32, device=device) * 100.0
        res.flags = torch.randint(
            low=0, high=16, size=(100,), dtype=torch.uint8, device=device
        )
        res.surface_type = torch.randint(
            low=0, high=4, size=(100,), dtype=torch.int32, device=device
        )
        res.inclination_deg = torch.rand(100, device=device) * 90.0
        res.roughness = torch.rand(100, device=device)
        res.frame_id = "random"
        res.stamp = float(np.random.rand())
        return res

    def __eq__(self, other):
        if not isinstance(other, VoxelCostFieldTorch):
            return NotImplemented
        if self.frame_id != other.frame_id or abs(self.stamp - other.stamp) > 1e-8:
            return False
        float_fields = (
            "origin",
            "length",
            "resolution",
            "cost",
            "inclination_deg",
            "roughness",
        )
        integer_fields = ("N", "grid_indices", "flags", "surface_type")
        return all(
            torch.allclose(getattr(self, name), getattr(other, name))
            for name in float_fields
        ) and all(
            torch.equal(getattr(self, name), getattr(other, name))
            for name in integer_fields
        )

    def to(self, device):
        self.device = device
        for attr in (
            "origin", "length", "resolution", "N",
            "grid_indices", "cost", "flags", "surface_type", "inclination_deg",
            "roughness",
        ):
            t = getattr(self, attr)
            if isinstance(t, torch.Tensor):
                setattr(self, attr, t.to(device))
        return self

    def __repr__(self):
        return (
            f"VoxelCostFieldTorch(M={self.cost.shape[0]}, frame={self.frame_id}, "
            f"device={self.device})"
        )
