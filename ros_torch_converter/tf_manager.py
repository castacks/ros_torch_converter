import os
import yaml
import torch
import numpy as np
from collections import defaultdict, deque
import sys

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from ros_torch_converter.datatypes.transform import TransformTorch

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.geometry_utils import TrajectoryInterpolator, pose_to_htm, htm_to_pose

class TransformData:
    """
    General class for storing transform information
    """
    def __init__(self, is_static, transforms, times):
        """
        Args:
            is_static: whether the tf changes over time
            transforms: [Tx7] array containing the transform from parent frame to frame
            times: [T] array containing times for transforms
        """
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
            self.t_min = times.min() - self.interp._tol
            self.t_max = times.max() + self.interp._tol

    def get_transform(self, t):
        return self.transform if self.is_static else self.interp(t)

class TfEdge:
    def __init__(self, src_frame, dst_frame, transform_data, invert=False):
        self.src_frame = src_frame
        self.dst_frame = dst_frame
        self.transform_data = transform_data
        self.invert = invert

    def get_transform(self, t):
        transform = self.transform_data.get_transform(t)

        if self.invert:
            T = pose_to_htm(transform)
            return htm_to_pose(np.linalg.inv(T))

        return transform

    def __repr__(self):
        return "{}->{} (static={})".format(self.src_frame, self.dst_frame, self.transform_data.is_static)

class TfGraph:
    # transforms only stores ROS TF transforms in their original direction 
    # (child_frame_id -> header.frame_id)
    # Forward/reverse graph edges are created internally by TfGraph
    def __init__(self, transforms):
        self.graph = defaultdict(dict)

        for src_frame, dst_frames in transforms.items():
            for dst_frame, transform_dict in dst_frames.items():
                transform_data=TransformData(
                    is_static=transform_dict['is_static'],
                    transforms=transform_dict['transforms'],
                    times=transform_dict['times']
                )
                self._add_edges(src_frame, dst_frame, transform_data)

    def _add_edges(self, src_frame, dst_frame, transform_data):
        self.graph[src_frame][dst_frame] = TfEdge(
            src_frame=src_frame,
            dst_frame=dst_frame,
            transform_data=transform_data,
            invert=False
        )

        self.graph[dst_frame][src_frame] = TfEdge(
            src_frame=dst_frame,
            dst_frame=src_frame,
            transform_data=transform_data,
            invert=True
        )

    def add_tf(self, src_frame, dst_frame, transforms, times):
        data = TransformData(
            is_static=False, 
            transforms=transforms, 
            times=times
        )

        self._add_edges(src_frame, dst_frame, data)

    def add_static_tf(self, src_frame, dst_frame, transform):
        data = TransformData(
            is_static=True,
            transforms=[transform],
            times=None
        )

        self._add_edges(src_frame, dst_frame, data)

    def has_transform(self, src_frame, dst_frame):
        return dst_frame in self.graph[src_frame]
    
    def is_transform_static(self, src_frame, dst_frame):
        return self.graph[src_frame][dst_frame].transform_data.is_static
    
    def get_edge(self, src_frame, dst_frame):
        return self.graph[src_frame][dst_frame]

    def _cycle_checker(self, frame, visited, parent):
        visited.add(frame)

        for neighbor in self.graph[frame]:
            if neighbor in visited:
                # cycle exists if this neighbor node is not 
                # the parent of the current node 
                # (the node that invoked the exploration of this node)
                if parent[frame] != neighbor:
                    print(f"Edge going from {frame}->{neighbor} creates a cycle!")
                    return True
            else:
                parent[neighbor] = frame
                has_cycle = self._cycle_checker(neighbor, visited, parent)
                if has_cycle:
                    return True
        return False

    def validate_graph(self):
        visited = set()
        parent = defaultdict(str)

        for frame in self.graph:
            if frame not in visited:
                parent[frame] = None
                has_cycle = self._cycle_checker(frame, visited, parent)
                if has_cycle:
                    return True
        print("tf graph is valid (no cycles detected)")
        return False

    def create_tree_with_root(self, root_frame_id):
        # will store list of TfNodes for creating TfTree with
        nodes = []

        visited = set()
        queue = deque()

        queue.append(root_frame_id)
        visited.add(root_frame_id)
 
        while queue:
            curr = queue.popleft()

            for neighbor in self.graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

                    edge = self.graph[curr][neighbor]

                    # See Slite documentation for explanation of TfEdge to TfNode conversion
                    node = TfNode(
                        frame_id=neighbor,
                        parent_frame_id=curr,
                        transform_data=edge.transform_data,
                        invert=not edge.invert
                    )
                    nodes.append(node)
        
        tree = TfTree(nodes=nodes)
        return tree

    def __repr__(self):
        ret_string = "Stored transforms:\n"
        for src in self.graph:
            for dst in self.graph[src]:
                ret_string += str(self.graph[src][dst]) + "\n"
        return ret_string

class TfNode:
    """
    Node for a transform in a tf tree
    """
    def __init__(self, frame_id, parent_frame_id, transform_data, invert=False, depth=-1):
        """
        Args:
            frame_id: frame id of the node
            parent_frame_id: frame id of the node's parent
            transform_data: TransformData object storing transform information for this TfNode
            invert: whether or not this node's transform is the inverse of the data stored in self.transform_data
        """
        self.frame_id = frame_id
        self.parent_frame_id = parent_frame_id
        self.transform_data = transform_data
        self.invert = invert        
        self.depth = depth

    def dummy_node(fid, depth=-1):
        I = np.array([[0., 0., 0., 0., 0., 0., 1.]])
        transform_data = TransformData(
            is_static=True,
            transforms=I,
            times=None
        )
        return TfNode(fid, "/ROOT", transform_data, False, depth)

    def get_transform(self, t):
        """
        Get the transform from parent_frame_id to frame_id at time t
        """
        transform = self.transform_data.get_transform(t)

        if self.invert:
            T = pose_to_htm(transform)
            return htm_to_pose(np.linalg.inv(T))

        return transform

    def __repr__(self):
        return "{}->{} (static={})".format(self.parent_frame_id, self.frame_id, self.transform_data.is_static)

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

        transform_data = TransformData(
            is_static=True,
            transforms=[transform],
            times=None
        )
        node = TfNode(
            frame_id=frame_id, 
            parent_frame_id=parent_frame_id, 
            transform_data=transform_data
        )

        self.nodes[frame_id] = node
        self.recompute_depth()

        return True

    def add_tf(self, frame_id, parent_frame_id, transforms, times):
        if frame_id in self.nodes.keys():
            curr_parent_frame_id = self.nodes[frame_id].parent_frame_id
            if curr_parent_frame_id != parent_frame_id:
                print('warning: overwriting tf {}->{} to {}->{}'.format(
                    curr_parent_frame_id, frame_id, parent_frame_id, frame_id
                ))
            
        transform_data = TransformData(
            is_static=False,
            transforms=transforms,
            times=times
        )
        node = TfNode(
            frame_id=frame_id, 
            parent_frame_id=parent_frame_id,
            transform_data=transform_data
        )
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

    def has_frame(self, x):
        if x in self.nodes.keys():
            return True
        
        parent_frames = {node.parent_frame_id for node in self.nodes.values()}
        return x in parent_frames

class TfManager:
    """
    Class that enables tf stuff for offline proc
    Essentially, this class will provide the same functionality as
        tf2.transform_listener, but with the kitti datasets
    """
    def __init__(self, device):
        self.tf_tree = TfTree(nodes=[])
        self.tf_graph = TfGraph(transforms=defaultdict(dict))
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def update_from_calib_config(self, calib_config):
        for calib_tf in calib_config['transform_params']:
            src_frame = calib_tf['to_frame']
            dst_frame = calib_tf['from_frame']

            if self.tf_graph.has_transform(src_frame, dst_frame):
                if not self.tf_graph.is_transform_static(src_frame, dst_frame):
                    print('tf {}->{} or {}->{} is not static. Skipping...'.format(src_frame, dst_frame, dst_frame, src_frame))
                    continue

                print('updating tf {}->{}'.format(src_frame, dst_frame))
                transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                self.add_static_tf(
                    src_frame, 
                    dst_frame, 
                    transform
                )

            else:
                print('couldnt find tf {}->{} and its inverse in tf graph! Adding...'.format(src_frame, dst_frame))
                transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                self.add_static_tf(
                    src_frame,
                    dst_frame,
                    transform
                )
        
        # re-validate graph
        print("Validating tf graph for cycles...")
        
        has_cycle = self.tf_graph.validate_graph()
        if has_cycle:
            sys.exit("""Error: tf graph has cycles! 
                        Transform lookups will not work properly, 
                        please check your transforms for loops. Exiting...""")

        # re-initialize tf_tree
        self.tf_tree = self.tf_graph.create_tree_with_root("vehicle")

    def add_static_tf(self, src_frame, dst_frame, transform):
        return self.tf_graph.add_static_tf(src_frame=src_frame, dst_frame=dst_frame, transform=transform)

    def add_tf(self, src_frame, dst_frame, transforms, times):
        return self.tf_graph.add_tf(src_frame=src_frame, dst_frame=dst_frame, transforms=transforms, times=times)

    def to_kitti(self, run_dir):
        base_dir = os.path.join(run_dir, 'tf')

        metadata = {"transforms": []}

        for src_frame in self.tf_graph.graph:
            for dst_frame in self.tf_graph.graph[src_frame]:
                edge = self.tf_graph.get_edge(src_frame, dst_frame)

                # only need to store the transform in one direction, 
                # since the other direction is just the inverse
                if not edge.invert:
                    metadata["transforms"].append({
                        'src_frame': src_frame,
                        'dst_frame': dst_frame,
                        'static': edge.transform_data.is_static
                    })

                    save_fp = os.path.join(base_dir, "{}_to_{}".format(
                        edge.src_frame.replace('/', '-'),
                        edge.dst_frame.replace('/', '-')
                    ))

                    os.makedirs(save_fp, exist_ok=True)

                    if edge.transform_data.is_static:
                        np.savetxt(os.path.join(save_fp, "static_transform.txt"), edge.transform_data.transform)
                    else:
                        np.savetxt(os.path.join(save_fp, "timestamps.txt"), edge.transform_data.times)
                        np.savetxt(os.path.join(save_fp, "transforms.txt"), edge.transform_data.transforms)

        with open(os.path.join(base_dir, "metadata.yaml"), 'w') as f:
            yaml.dump(metadata, f)

    def from_kitti(run_dir, device='cpu'):
        tf_manager = TfManager(device)
        base_dir = os.path.join(run_dir, 'tf')
        metadata_fp = os.path.join(base_dir, 'metadata.yaml')

        metadata = yaml.safe_load(open(metadata_fp, 'r'))

        all_transforms = defaultdict(dict)

        for transform_metadata in metadata["transforms"]:
            transform_dir = os.path.join(base_dir, "{}_to_{}".format(
                transform_metadata["src_frame"].replace('/', '-'),
                transform_metadata["dst_frame"].replace('/', '-')
            ))

            dst_frame = transform_metadata["dst_frame"]
            src_frame = transform_metadata["src_frame"]
            is_static = transform_metadata["static"]

            if is_static:
                transforms = np.loadtxt(os.path.join(transform_dir, "static_transform.txt")).reshape(1, 7)
                timestamps = np.zeros(1)
            else:
                transforms = np.loadtxt(os.path.join(transform_dir, "transforms.txt"))
                timestamps = np.loadtxt(os.path.join(transform_dir, "timestamps.txt"))

            all_transforms[src_frame][dst_frame] = {
                'src_frame': src_frame,
                'dst_frame': dst_frame,
                'is_static': is_static,
                'transforms': transforms,
                'times': timestamps
            }

        tf_manager.tf_graph = TfGraph(all_transforms)
        
        print("Validating tf graph for cycles...")
        
        has_cycle = tf_manager.tf_graph.validate_graph()
        if has_cycle:
            sys.exit("""Error: tf graph has cycles! 
                        Transform lookups will not work properly, 
                        please check your transforms for loops. Exiting...""")

        tf_manager.tf_tree = tf_manager.tf_graph.create_tree_with_root("vehicle")
    
        return tf_manager

    def from_rosbag(rosbag_fp, use_bag_time=False, dt=0.1, device='cpu'):
        tf_manager = TfManager(device)

        bag_fps = sorted([x for x in os.listdir(rosbag_fp) if '.mcap' in x])

        # adjacency list of (parent_frame_id, list of transforms with this frame as parent_frame_id)
        transforms = defaultdict(dict)

        bagpath = Path(rosbag_fp)

        typestore = get_typestore(Stores.ROS2_HUMBLE)

        with AnyReader([bagpath], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic in ['/tf', '/tf_static']]

            cnt = 1

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                topic = connection.topic

                for tf_msg in msg.transforms:
                    src_frame = tf_msg.child_frame_id
                    dst_frame = tf_msg.header.frame_id
                    t = stamp_to_time(tf_msg.header.stamp)

                    if dst_frame not in transforms[src_frame]:
                        transforms[src_frame][dst_frame] = {
                            'src_frame': src_frame,
                            'dst_frame': dst_frame,
                            'is_static': topic == '/tf_static',
                            'transforms': np.zeros([0, 7]),
                            'times': np.zeros(0)
                        }
                    
                    if dt > 0. and len(transforms[src_frame][dst_frame]['times']) > 0:
                        if t - transforms[src_frame][dst_frame]['times'][-1] < dt:
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

                    transforms[src_frame][dst_frame]['times'] = np.append(transforms[src_frame][dst_frame]['times'], t)
                    transforms[src_frame][dst_frame]['transforms'] = np.append(transforms[src_frame][dst_frame]['transforms'], tf_data.reshape(1,7), axis=0)
                    
                cnt += 1

        tf_manager.tf_graph = TfGraph(transforms)
        
        print("Validating tf graph for cycles...")
        
        has_cycle = tf_manager.tf_graph.validate_graph()
        if has_cycle:
            sys.exit("""Error: tf graph has cycles! 
                        Transform lookups will not work properly, 
                        please check your transforms for loops. Exiting...""")
        
        tf_manager.tf_tree = tf_manager.tf_graph.create_tree_with_root("vehicle")
    
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
            tmin = max([node.transform_data.t_min for node in all_tfs])
            tmax = min([node.transform_data.t_max for node in all_tfs])

            return tmin, tmax
        else:
            return float('inf'), -float('inf')

    def get_valid_times_from_list(self, frame_list):
        """
        get valid sample times for a list of frames
        """
        tmin = -float('inf')
        tmax = float('inf')

        frame_list = [x for x in frame_list if self.tf_tree.has_frame(x)]

        for frame in frame_list:
            _tmin, _tmax = self.get_valid_times(frame, frame_list[0])
            tmin = max(tmin, _tmin)
            tmax = min(tmax, _tmax)

        return tmin, tmax

    def can_transform(self, src_frame, dst_frame, t):
        tmin, tmax = self.get_valid_times(src_frame, dst_frame)
        return t >= tmin and t <= tmax

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