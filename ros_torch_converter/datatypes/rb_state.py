import os
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from ros_torch_converter.utils import update_frame_file, update_timestamp_file, read_frame_file, read_timestamp_file

from nav_msgs.msg import Odometry

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class OdomRBStateTorch(TorchCoordinatorDataType):
    """state as [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    """
    to_rosmsg_type = Odometry
    from_rosmsg_type = Odometry
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.child_frame_id = ""
        self.state = torch.zeros(13, device=device)
        self.device = device

    def from_rosmsg(msg, device='cpu'):
        res = OdomRBStateTorch(device=device)

        res.state = torch.tensor([
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
            msg.twist.twist.angular.z
        ], device=device).float()
        
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        res.child_frame_id = msg.child_frame_id
        return res

    def from_numpy(data, child_frame_id, device='cpu'):
        return OdomRBStateTorch.from_torch(torch.tensor(data, dtype=torch.float, device=device), child_frame_id)

    def from_torch(data, child_frame_id):
        res = OdomRBStateTorch(device=data.device)
        res.state = data
        res.child_frame_id = child_frame_id

        return res

    def to_rosmsg(self):
        msg = Odometry()
        msg.header.stamp = time_to_stamp(self.time)
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        msg.pose.pose.position.x = self.state[0].item()
        msg.pose.pose.position.y = self.state[1].item()
        msg.pose.pose.position.z = self.state[2].item()

        msg.pose.pose.orientation.x = self.state[3].item()
        msg.pose.pose.orientation.y = self.state[4].item()
        msg.pose.pose.orientation.z = self.state[5].item()
        msg.pose.pose.orientation.w = self.state[6].item()

        msg.twist.twist.linear.x = self.state[7].item()
        msg.twist.twist.linear.y = self.state[8].item()
        msg.twist.twist.linear.z = self.state[9].item()

        msg.twist.twist.angular.x = self.state[10].item()
        msg.twist.twist.angular.y = self.state[11].item()
        msg.twist.twist.angular.z = self.state[12].item()

        return msg

    def to(self, device):
        self.device = device
        self.state = self.state.to(device)
        return self

    def to_kitti(self, base_dir, idx):
        """
        note that some dtypes  should be stored as rows of a matrix
        """
        update_timestamp_file(base_dir, idx, self.stamp)
        update_frame_file(base_dir, idx, 'frame_id', self.frame_id)
        update_frame_file(base_dir, idx, 'child_frame_id', self.child_frame_id)

        save_fp = os.path.join(base_dir, "data.txt")
        if not os.path.exists(save_fp):
            data = float('inf') * np.ones([idx+1, 13])
        else:
            #need to reshape for 1-row data
            data = np.loadtxt(save_fp).reshape(-1, 13)

        if data.shape[0] < (idx+1):
            data_new = float('inf') * np.ones([idx+1, 13])
            data_new[:data.shape[0]] = data
            data = data_new

        data[idx] = self.state.cpu().numpy()

        np.savetxt(save_fp, data)

    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "data.txt")
        state = np.loadtxt(fp).reshape(-1, 13)[idx]
        state = torch.tensor(state, dtype=torch.float, device=device)

        child_frame_id = read_frame_file(base_dir, idx, 'child_frame_id')

        rbst = OdomRBStateTorch.from_torch(state, child_frame_id)

        rbst.stamp = read_timestamp_file(base_dir, idx)
        rbst.frame_id = read_frame_file(base_dir, idx, 'frame_id')

        return rbst
    
    def from_kitti_multi(base_dir, idxs, device='cpu'):
        fp = os.path.join(base_dir, "data.txt")
        states = np.loadtxt(fp).reshape(-1, 13)[idxs]
        states = torch.tensor(states, dtype=torch.float, device=device)

        stamps = read_timestamp_file(base_dir, idxs)
        frame_id = read_frame_file(base_dir, idxs[0], 'frame_id')
        child_frame_id = read_frame_file(base_dir, idxs[0], 'child_frame_id')

        rbsts = [OdomRBStateTorch.from_torch(state, child_frame_id) for state in states]

        return rbsts

    def rand_init(device='cpu'):
        x = torch.rand(13)

        angs = np.random.rand(3) * 2*np.pi
        q = R.from_euler('xyz', angs).as_quat()
        q = torch.tensor(q, dtype=torch.float, device=device)
        x[3:7] = q

        rbst = OdomRBStateTorch.from_torch(x, 'random_child')
        rbst.frame_id = 'random'
        rbst.stamp = np.random.rand()

        return rbst

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if self.child_frame_id != other.child_frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not torch.allclose(self.state, other.state):
            return False

        return True

    def __repr__(self):
        return "OdomRBStateTorch from {} to {} with x:\n{} (time = {:.2f}, device = {})".format(self.frame_id, self.child_frame_id, self.state.cpu().numpy().round(4), self.stamp, self.device)
    