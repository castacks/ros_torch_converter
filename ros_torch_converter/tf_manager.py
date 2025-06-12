import os
import yaml
import torch
import numpy as np

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from ros_torch_converter.datatypes.transform import TransformTorch

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.geometry_utils import TrajectoryInterpolator, pose_to_htm

class TfNode:
    """
    Node for a transform in a tf tree
    """
    def __init__(self, frame_id, parent_frame_id, transforms, times, is_static, depth=-1):
        """
        Args:
            frame_id: frame id of the node
            parent_frame_id: frame id of the node's parent
            transforms: [Tx7] array containing the transform from parent frame to frame
            times: [T] array containing times for transforms
            is_static: whether the tf changes over time
        """
        self.frame_id = frame_id
        self.parent_frame_id = parent_frame_id
        self.is_static = is_static

        if self.is_static:
            self.transform = transforms[0]
            self.t_min = -float('inf')
            self.t_max = float('inf')
        else:
            idxs = np.argsort(times)

            self.times = times[idxs]
            self.transforms = transforms[idxs]

            self.interp = TrajectoryInterpolator(self.times, self.transforms)
            self.t_min = times.min()
            self.t_max = times.max()

        self.depth = depth

    def dummy_node(fid, depth=-1):
        I = np.array([[0., 0., 0., 0., 0., 0., 1.]])
        return TfNode(fid, "/ROOT", I, None, True, depth)

    def get_transform(self, t):
        """
        Get the transform from parent_frame_id to frame_id at time t
        """
        return self.transform if self.is_static else self.interp(t)

    def __repr__(self):
        return "{}->{} (static={})".format(self.parent_frame_id, self.frame_id, self.is_static)

class TfTree:
    """
    The actual tf tree(s)
    """
    def __init__(self, nodes):
        self.nodes = {x.frame_id:x for x in nodes}

        self.recompute_depth()

    def recompute_depth(self):
        #compute node depths (just have every node iterate up to its root)
        for node in self.nodes.values():
            node.depth = -1

        self.roots = set()

        for node in self.nodes:
            branch = self.get_branch(node)
            for i, bnode in enumerate(branch):
                if bnode.depth == -1:
                    bnode.depth = i
                else:
                    assert i == bnode.depth, "uh oh tree depth is bad"

            # if branch[0].parent_frame_id != "/ROOT":
            self.roots.add(branch[0].parent_frame_id)

        # for root in self.roots:
        #     self.nodes[root] = TfNode.dummy_node(root)

    def add_static_tf(self, frame_id, parent_frame_id, transform):
        if frame_id in self.nodes.keys():
            curr_parent_frame_id = self.nodes[frame_id].parent_frame_id
            if curr_parent_frame_id != parent_frame_id:
                print('warning: overwriting tf {}->{} to {}->{}'.format(
                    curr_parent_frame_id, frame_id, parent_frame_id, frame_id
                ))

        node = TfNode(frame_id=frame_id, parent_frame_id=parent_frame_id, transforms=[transform], times=None, is_static=True)

        self.nodes[frame_id] = node
        self.recompute_depth()

        return True
            
    def get_branch(self, frame_id):
        if isinstance(frame_id, TfNode):
            frame_id = frame_id.frame_id

        if frame_id in self.roots:
            return []

        curr_node = self.nodes[frame_id]
        branch = [curr_node]

        while curr_node.parent_frame_id in self.nodes.keys() and curr_node.frame_id != curr_node.parent_frame_id:
            curr_node = self.nodes[curr_node.parent_frame_id]
            branch.insert(0, curr_node)

        return branch

    def get_lca_paths(self, frame1, frame2):
        """
        Get paths to lowest common ancestor for frame1, frame2

        Return None if none exist
        """
        branch1 = self.get_branch(frame1)
        branch2 = self.get_branch(frame2)

        #LCA doesnt exist iff. roots are different
        branch1_root_fid = branch1[0].parent_frame_id if len(branch1) > 0 else frame1
        branch2_root_fid = branch2[0].parent_frame_id if len(branch2) > 0 else frame2

        if branch1_root_fid != branch2_root_fid:
            return None

        depth = 0
        for b1n, b2n in zip(branch1, branch2):
            if b1n.frame_id != b2n.frame_id:
                break
            else:
                depth += 1

        return branch1[depth:], branch2[depth:]

    def __repr__(self):
        """
        This gets a bit messy because we only have parent pointers
        """
        out = ""
        for root in self.roots:
            out += self._repr_helper(root, depth=0)
        return out

    def _repr_helper(self, frame_id, depth):
        out = '- ' * depth + frame_id + '\n'
        for node in self.nodes.values():
            if node.parent_frame_id == frame_id:
                out += self._repr_helper(node.frame_id, depth+1)
        return out

class TfManager:
    """
    Class that enables tf stuff for offline proc
    Essentially, this class will provide the same functionality as
        tf2.transform_listener, but with the kitti datasets
    """
    def __init__(self, device):
        self.device = device

    def to(self, device):
        self.device = device

    def update_from_calib_config(self, calib_config):
        for calib_tf in calib_config['transform_params']:
            src_frame = calib_tf['from_frame']
            dst_frame = calib_tf['to_frame']

            if dst_frame in self.tf_tree.nodes.keys():
                tf_node = self.tf_tree.nodes[dst_frame]

                if not tf_node.is_static:
                    print('tf {}->{} is not static. Skipping...'.format(src_frame, dst_frame))
                    continue

                if tf_node.parent_frame_id != src_frame and tf_node.parent_frame_id != "/ROOT":
                    print('got tf {}->{} in calib, but is {}->{} in data. Skipping...'.format(src_frame, dst_frame, tf_node.parent_frame_id, dst_frame))
                else:
                    print('updating tf {}->{}'.format(src_frame, dst_frame))
                    transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                    res = self.add_static_tf(src_frame, dst_frame, transform)

            else:
                print('couldnt find tf {}->{} in tf tree! Adding...'.format(src_frame, dst_frame))
                transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                res = self.add_static_tf(src_frame, dst_frame, transform)

                if not res:
                    print('couldnt add tf!')

    def add_static_tf(self, src_frame, dst_frame, transform):
        return self.tf_tree.add_static_tf(parent_frame_id=src_frame, frame_id=dst_frame, transform=transform)

    def to_kitti(self, run_dir):
        base_dir = os.path.join(run_dir, 'tf')

        metadata = {"frames": []}

        for node in self.tf_tree.nodes.values():
            if node.parent_frame_id == "/ROOT":
                continue

            metadata["frames"].append({
                'frame': node.frame_id,
                'parent': node.parent_frame_id,
                'static': node.is_static
            })

            save_fp = os.path.join(base_dir, "{}_to_{}".format(
                    node.parent_frame_id.replace('/', '-'),
                    node.frame_id.replace('/', '-')
            ))

            os.makedirs(save_fp, exist_ok=True)

            if node.is_static:
                np.savetxt(os.path.join(save_fp, "static_transform.txt"), node.transform)
            else:
                np.savetxt(os.path.join(save_fp, "timestamps.txt"), node.times)
                np.savetxt(os.path.join(save_fp, "transforms.txt"), node.transforms)

        yaml.dump(metadata, open(os.path.join(base_dir, "metadata.yaml"), 'w'))

    def from_kitti(run_dir, device='cpu'):
        tf_manager = TfManager(device)
        base_dir = os.path.join(run_dir, 'tf')
        metadata_fp = os.path.join(base_dir, 'metadata.yaml')

        metadata = yaml.safe_load(open(metadata_fp, 'r'))

        frames = {}

        for frame_metadata in metadata["frames"]:
            frame_dir = os.path.join(base_dir, "{}_to_{}".format(
                    frame_metadata["parent"].replace('/', '-'),
                    frame_metadata["frame"].replace('/', '-')
            ))

            dst_frame = frame_metadata["frame"]
            src_frame = frame_metadata["parent"]
            is_static = frame_metadata["static"]

            if is_static:
                transforms = np.loadtxt(os.path.join(frame_dir, "static_transform.txt")).reshape(1, 7)
                timestamps = np.zeros(1)
            else:
                transforms = np.loadtxt(os.path.join(frame_dir, "transforms.txt"))
                timestamps = np.loadtxt(os.path.join(frame_dir, "timestamps.txt"))

            frames[dst_frame] = {
                'frame_id': dst_frame,
                'parent_frame_id': src_frame,
                'is_static': is_static,
                'transforms': transforms,
                'times': timestamps
            }

        tf_manager.tf_tree = TfTree(nodes=[TfNode(**v) for v in frames.values()])
    
        return tf_manager

    def from_rosbag(rosbag_fp, use_bag_time=False, dt=0.1, device='cpu'):
        tf_manager = TfManager(device)

        bag_fps = sorted([x for x in os.listdir(rosbag_fp) if '.mcap' in x])

        #have every frame keep track of tf to its parent
        frames = {}

        bagpath = Path(rosbag_fp)

        typestore = get_typestore(Stores.ROS2_HUMBLE)

        with AnyReader([bagpath], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic in ['/tf', '/tf_static']]

            cnt = 1

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                topic = connection.topic

                for tf_msg in msg.transforms:
                    src_frame = tf_msg.header.frame_id
                    dst_frame = tf_msg.child_frame_id
                    t = stamp_to_time(tf_msg.header.stamp)

                    if dst_frame not in frames.keys():
                        frames[dst_frame] = {
                            'frame_id': dst_frame,
                            'parent_frame_id': src_frame,
                            'is_static': topic == '/tf_static',
                            'transforms': np.zeros([0, 7]),
                            'times': np.zeros(0)
                        }
                    else:
                        assert src_frame == frames[dst_frame]['parent_frame_id'], "TfManager does not currently support rewiring the tf tree"

                    if dt > 0. and len(frames[dst_frame]['times']) > 0:
                        if t - frames[dst_frame]['times'][-1] < dt:
                            continue

                    tf_data = np.array([
                        tf_msg.transform.translation.x,
                        tf_msg.transform.translation.y,
                        tf_msg.transform.translation.z,
                        tf_msg.transform.rotation.x,
                        tf_msg.transform.rotation.y,
                        tf_msg.transform.rotation.z,
                        tf_msg.transform.rotation.w
                    ])

                    frames[dst_frame]['times'] = np.append(frames[dst_frame]['times'], t)
                    frames[dst_frame]['transforms'] = np.append(frames[dst_frame]['transforms'], tf_data.reshape(1,7), axis=0)
                    
                cnt += 1

        tf_manager.tf_tree = TfTree(nodes=[TfNode(**v) for v in frames.values()])
    
        return tf_manager

    def get_valid_times(self, frame1, frame2):
        """
        Get the range of times that we can transform between frame1 and frame2
        """
        if frame1 == frame2:
            return -float('inf'), float('inf')

        lca_paths = self.tf_tree.get_lca_paths(frame1, frame2)
        if lca_paths:
            all_tfs = lca_paths[0] + lca_paths[1]
            tmin = max([node.t_min for node in all_tfs])
            tmax = min([node.t_max for node in all_tfs])

            return tmin, tmax
        else:
            return float('inf'), -float('inf')

    def get_valid_times_from_list(self, frame_list):
        """
        get valid sample times for a list of frames
        """
        tmin = -float('inf')
        tmax = float('inf')

        for frame in frame_list:
            _tmin, _tmax = self.get_valid_times(frame, frame_list[0])
            tmin = max(tmin, _tmin)
            tmax = min(tmax, _tmax)

        return tmin, tmax

    def can_transform(self, src_frame, dst_frame, t):
        tmin, tmax = self.get_valid_times(src_frame, dst_frame)
        return t > tmin and t < tmax

    def get_transform(self, frame1, frame2, t):
        """
        Get the transform from frame1 to frame2 at time t
        """
        frame1_path, frame2_path = self.tf_tree.get_lca_paths(frame1, frame2)

        tf = torch.eye(4, device=self.device)
        for node in reversed(frame1_path):
            new_tf = torch.tensor(pose_to_htm(node.get_transform(t)), dtype=torch.float, device=self.device)
            tf = tf @ torch.linalg.inv(new_tf)

        for node in frame2_path:
            new_tf = torch.tensor(pose_to_htm(node.get_transform(t)), dtype=torch.float, device=self.device)
            tf = tf @ new_tf

        tf = TransformTorch.from_torch(tf, child_frame_id=frame2)
        tf.frame_id = frame1
        tf.stamp = t

        return tf

if __name__ == '__main__':
    import time
    import yaml

    bag_fp = "/media/striest/offroad/rosbags/20250508/teleop/power_tower_sidehill6/"
    calib_fp = "/home/tartandriver/tartandriver_ws/src/core/static_tf_publisher/config/offroad/yamaha.yaml"

    calib_config = yaml.safe_load(open(calib_fp, 'r'))

    kitti_fp = '/home/tartandriver/workspace/aaa'

    # tf_manager = TfManager.from_rosbag(bag_fp, device='cuda')
    # tf_manager.to_kitti(kitti_fp)

    tf_manager = TfManager.from_kitti(kitti_fp, device='cuda')
    tf_manager.update_from_calib_config(calib_config)

    print(tf_manager.tf_tree)

    src_frame = 'sensor_init'
    dst_frame = 'thermal_left/camera_link'
    # dst_frame = 'thermal_left/optical_frame'

    trange = tf_manager.get_valid_times(src_frame, dst_frame)

    if trange[0] < 0:
        trange = (0., 100.)

    ts = np.arange(trange[0], trange[1], 0.5)

    torch.set_printoptions(sci_mode=False, precision=3)

    print('running...')

    for t in ts:
        print('{}->{} @ t={:.2f}'.format(src_frame, dst_frame, t))
        t1 = time.time()
        tf = tf_manager.get_transform(src_frame, dst_frame, t)
        torch.cuda.synchronize()
        t2 = time.time()
        print(tf)

        print('took {:.4f}s'.format(t2-t1))
        break