import os
import yaml
import torch
import numpy as np

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from ros_torch_converter.datatypes.transform import TransformTorch

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.geometry_utils import TrajectoryInterpolator, pose_to_htm, htm_to_pose

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
            self.t_min = times.min() - self.interp._tol
            self.t_max = times.max() + self.interp._tol

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

    def add_tf(self, frame_id, parent_frame_id, transforms, times):
        if frame_id in self.nodes.keys():
            curr_parent_frame_id = self.nodes[frame_id].parent_frame_id
            if curr_parent_frame_id != parent_frame_id:
                print('warning: overwriting tf {}->{} to {}->{}'.format(
                    curr_parent_frame_id, frame_id, parent_frame_id, frame_id
                ))
            
        node = TfNode(frame_id=frame_id, parent_frame_id=parent_frame_id, transforms=transforms, times=times, is_static=False)
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
    Class that enables tf stuff for offline proc.

    Essentially, this class provides the same functionality as tf2's transform_listener,
    but for offline kitti-style datasets -- with one important difference: tf2 assumes a
    single-rooted tree, but we frequently have *multiple* independent odometry sources
    (e.g. super odometry rooted at `sensor_init`, RTK rooted at `earth`), each of which is
    only sensibly comparable to the vehicle's own sensor tree, not to each other.

    This is handled by keeping:
      - one "vehicle" tree: all transforms between sensor frames, rooted at the vehicle frame
      - one TfTree per odometry source in `odom_trees`, each rooted at that source's own
        global frame (e.g. `earth`, `sensor_init`)
      - one identity "bridge" node per odometry source in bridges, connecting that
        source's vehicle-frame leaf (e.g. vehicle_rtk) to the root of the vehicle tree

    A lookup between two frames finds whichever tree(s) contain them:
      - both frames live only in vehicle_tree                -> lookup directly in vehicle_tree
      - one/both frames live in the same odom source's tree    -> lookup in that odom source's
                                                                    tree merged with the bridge +
                                                                    vehicle_tree
      - the frames live in two different odom sources' trees -> not supported; there's no
        principled way to connect two independent global references through a single vehicle
        pose estimate
    """
    def __init__(self, device):
        self.vehicle_tree = TfTree(nodes=[])
        self.odom_trees = {}
        self.bridges = {}
        self.merged_tree_cache = {}
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def update_from_calib_config(self, calib_config, tree=None):
        tree = self.vehicle_tree if tree is None else tree

        for calib_tf in calib_config['transform_params']:
            src_frame = calib_tf['from_frame']
            dst_frame = calib_tf['to_frame']

            if dst_frame in tree.nodes.keys():
                tf_node = tree.nodes[dst_frame]

                if not tf_node.is_static:
                    print('tf {}->{} is not static. Skipping...'.format(src_frame, dst_frame))
                    continue
                    # print('tf {}->{} is not static. Overriding with calibration...'.format(src_frame, dst_frame))
                    # # Force override dynamic TF with static calibration to handle old thermal
                    # transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                    # res = self.add_static_tf(src_frame, dst_frame, transform)

                if tf_node.parent_frame_id != src_frame and tf_node.parent_frame_id != "/ROOT":
                    print('got tf {}->{} in calib, but is {}->{} in data. Skipping...'.format(src_frame, dst_frame, tf_node.parent_frame_id, dst_frame))
                else:
                    print('updating tf {}->{}'.format(src_frame, dst_frame))
                    transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                    res = tree.add_static_tf(parent_frame_id=src_frame, frame_id=dst_frame, transform=transform)
            else:
                print('couldnt find tf {}->{} in tf tree! Adding...'.format(src_frame, dst_frame))
                transform = np.array(calib_tf['translation'] + calib_tf['quaternion'])
                res = tree.add_static_tf(parent_frame_id=src_frame, frame_id=dst_frame, transform=transform)

                if not res:
                    print('couldnt add tf!')

        self.merged_tree_cache = {}

    def add_odom_tf(self, name, src_frame, dst_frame, transforms, times):
        """
        Add/replace dynamic tf data for one odometry source. src_frame is the
        odometry source's global root (e.g. earth), dst_frame is its vehicle-equivalent
        leaf frame (e.g. vehicle_rtk).
        """
        tree = self.odom_trees.setdefault(name, TfTree(nodes=[]))
        res = tree.add_tf(src_frame, dst_frame, transforms, times=times)
        self.merged_tree_cache.pop(name, None)
        return res

    def add_bridge(self, name, odom_vehicle_frame):
        """
        Add the identity bridge node connecting one odometry source's vehicle-equivalent
        leaf frame (e.g. vehicle_rtk) to the root of the vehicle tf tree (vehicle).
        """
        identity_pose = np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1]), axis=0)
        node = TfNode(frame_id="vehicle", parent_frame_id=odom_vehicle_frame,
                      transforms=torch.from_numpy(identity_pose).to(self.device), times=None, is_static=True)
        self.bridges[name] = node
        self.merged_tree_cache.pop(name, None)
        return node

    def build_bridges(self, odometry_configs):
        """
        Helper function: given the odometry_tf section of the ros2bag_2_kitti config, 
        add a bridge node for every entry that has been added to the dict of odom_trees
        """
        for cfg in odometry_configs:
            name = cfg['name']
            if name not in self.odom_trees or cfg['odometry']['bridge_frame'] not in self.odom_trees[name].nodes:
                print(f"No odometry data found for {name}, skipping bridge")
                continue
            self.add_bridge(name, cfg['odometry']['bridge_frame'])

    def get_full_tree(self, name):
        """
        Returns the full tree including the odometry source specified by the name parameter
        This is the tree formed by this odometry source's own tf tree + its bridge node + the full vehicle tree. 
        This is the tree that lookups involving that odometry source are actually done against.
        The cache of trees, merged_tree_cache, is updated if this lookup tree was not previously in the cache
        """
        assert name in self.merged_tree_cache or name in self.odom_trees, f"Unknown odometry source: {name}"

        if name not in self.merged_tree_cache:
            nodes = list(self.odom_trees[name].nodes.values())
            if name in self.bridges:
                nodes = nodes + [self.bridges[name]] + list(self.vehicle_tree.nodes.values())

            self.merged_tree_cache[name] = TfTree(nodes=nodes)

        return self.merged_tree_cache[name]

    def find_tree_for_frame(self, frame_id):
        """
        Returns a list of names of the tf tree(s) where frame_id is contained in,
        "vehicle" if it's only in the vehicle tf tree, and an empty list if it's
        not in any of the trees
        """
        odoms = [name for name, tree in self.odom_trees.items()
                 if frame_id in tree.nodes or tree.has_frame(frame_id)]
        if odoms:
            return odoms

        if frame_id in self.vehicle_tree.nodes or self.vehicle_tree.has_frame(frame_id):
            return ['vehicle']

        return []

    def has_frame(self, frame_id):
        return bool(self.find_tree_for_frame(frame_id))

    def get_lookup_tree(self, frame1, frame2):
        """
        Finds the tree to lookup the frame1<->frame2 transform from
        Valid pairs of frame lookups are:
        1. vehicle tree frame <-> vehicle tree frame
        2. vehicle tree frame <-> odometry specific tree frame

        A transform lookup is undefined if both frames are from different odometry specific trees,
        for example earth <-> sensor_init
        """
        loc1 = self.find_tree_for_frame(frame1)
        loc2 = self.find_tree_for_frame(frame2)

        assert loc1, f"Frame {frame1} not found in any tf tree"
        assert loc2, f"Frame {frame2} not found in any tf tree"

        odoms1 = [x for x in loc1 if x != 'vehicle']
        odoms2 = [x for x in loc2 if x != 'vehicle']

        if odoms1 and odoms2:
            common = set(odoms1) & set(odoms2)
            assert common, f"Cannot transform between {frame1} and {frame2} as they belong to different odometry sources, {odoms1} and {odoms2}"
            return self.get_full_tree(next(iter(common)))

        if odoms1:
            return self.get_full_tree(odoms1[0])
        if odoms2:
            return self.get_full_tree(odoms2[0])

        return self.vehicle_tree
    
    def node_to_kitti(self, node, tree_name, metadata, base_dir):
        metadata["frames"].append({
            'frame': node.frame_id,
            'parent': node.parent_frame_id,
            'static': node.is_static,
            'tree': tree_name,
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

    def to_kitti(self, run_dir):
        base_dir = os.path.join(run_dir, 'tf')

        metadata = {"frames": [], "tree_links": []}

        for node in self.vehicle_tree.nodes.values():
            if node.parent_frame_id == "/ROOT":
                continue
            self.node_to_kitti(node, 'vehicle', metadata, base_dir)

        for name, tree in self.odom_trees.items():
            for node in tree.nodes.values():
                if node.parent_frame_id == "/ROOT":
                    continue
                self.node_to_kitti(node, f"odom:{name}", metadata, base_dir)

        for name, node in self.bridges.items():
            metadata["tree_links"].append({
                'odom': name,
                'bridge_frame': node.parent_frame_id,
            })

        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "metadata.yaml"), 'w') as f:
            yaml.dump(metadata, f)

    def from_kitti(run_dir, device='cpu'):
        tf_manager = TfManager(device)
        base_dir = os.path.join(run_dir, 'tf')
        metadata_fp = os.path.join(base_dir, 'metadata.yaml')

        metadata = yaml.safe_load(open(metadata_fp, 'r'))

        vehicle_nodes = []
        odom_nodes = {}
        bridge_nodes = {}

        for frame_metadata in metadata["frames"]:
            frame_dir = os.path.join(base_dir, "{}_to_{}".format(
                frame_metadata["parent"].replace('/', '-'),
                frame_metadata["frame"].replace('/', '-')
            ))

            dst_frame = frame_metadata["frame"]
            src_frame = frame_metadata["parent"]
            is_static = frame_metadata["static"]
            tree_name = frame_metadata["tree"]

            if is_static:
                transforms = np.loadtxt(os.path.join(frame_dir, "static_transform.txt")).reshape(1, 7)
                timestamps = np.zeros(1)
            else:
                transforms = np.loadtxt(os.path.join(frame_dir, "transforms.txt"))
                timestamps = np.loadtxt(os.path.join(frame_dir, "timestamps.txt"))

            node = TfNode(frame_id=dst_frame, parent_frame_id=src_frame,
                          transforms=transforms, times=timestamps, is_static=is_static)

            assert tree_name == 'vehicle' or tree_name.startswith("odom:"), f"Tree name {tree_name} does not exist"

            if tree_name == 'vehicle':
                vehicle_nodes.append(node)
            elif tree_name.startswith('odom:'):
                odom_nodes.setdefault(tree_name.split(':', 1)[1], []).append(node)
        
        for tree_link_metadata in metadata["tree_links"]:
            odom = tree_link_metadata["odom"]
            bridge_frame = tree_link_metadata["bridge_frame"]
            tf_manager.add_bridge(odom, bridge_frame)

        tf_manager.vehicle_tree = TfTree(nodes=vehicle_nodes)
        tf_manager.odom_trees = {name: TfTree(nodes=nodes) for name, nodes in odom_nodes.items()}

        return tf_manager
 
    def normalize_odom_tfs(odom_frames, odometry_configs, tf_manager):
        odom_configs_by_name = {cfg["name"]: cfg for cfg in odometry_configs}

        # go through odom_nodes, if (parent, child) is in dynamic odom tfs then need to ensure that this transform expresses the pose of the vehicle frame
        # for example, with RTK odometry, the transform is earth -> gq7_imu_link but want the transform to be earth -> vehicle
        for tree_name, frames in odom_frames.items():
            cfg = odom_configs_by_name[tree_name]
            raw_child_frame = cfg['odometry']['child_frame']

            if raw_child_frame not in frames:
                # no transforms to normalize for this odometry source, skip
                continue

            odom_node = frames[raw_child_frame]

            if raw_child_frame != 'vehicle':
                assert not odom_node['is_static'], "Should not be normalizing a static tf"
                
                ts = frames[raw_child_frame]['times'][0]
                T_child_vehicle = tf_manager.get_transform_from_tree(raw_child_frame, 'vehicle', ts, tf_manager.vehicle_tree)

                # normalize transforms: convert (odom -> child) to (odom -> vehicle)
                # by applying static (child -> vehicle) transform
                normalized_transforms = []
                for pose in odom_node['transforms']:
                    T_odom_child = torch.from_numpy(pose_to_htm(pose)).float().to(tf_manager.device)
                    T_odom_vehicle = T_odom_child @ T_child_vehicle.transform
                    normalized_transforms.append(
                        htm_to_pose(T_odom_vehicle).cpu().numpy()
                    )
                
                odom_node['transforms'] = np.asarray(normalized_transforms)

            # replace child frame id with bridge frame id
            bridge_frame = cfg['odometry']['bridge_frame']
            odom_node['frame_id'] = bridge_frame
            odom_frames[tree_name][bridge_frame] = odom_node
            del odom_frames[tree_name][raw_child_frame]

    def from_rosbag(rosbag_fp, odometry_configs=None, tf_inversions=None, use_bag_time=False, dt=0.1, device='cpu'):
        """
        Args:
            odometry_configs: optional list of dicts, e.g. the odometry_tf section of
                                the yaml config
                Any raw (parent_frame_id -> child_frame_id) tf edge published on /tf or
                /tf_static matching one of these pairs is routed into its own odometry
                tree instead of the shared vehicle tree.
        """
        tf_manager = TfManager(device)
        odometry_configs = odometry_configs or []
        tf_inversions = tf_inversions or []

        # set of (src, dst) corresponding to parent/child pairs where warning for rewiring tf tree has been printed
        tf_rewire_warnings = set()

        # create mapping from tfs to odometry source name for determining tree membership for transforms
        tfs_to_odom = {}

        # create set of tfs to invert as a preprocessing step before inserting into tree
        tfs_to_invert = set()

        for cfg in odometry_configs:
            tfs_to_odom[(cfg['odometry']['parent_frame'], cfg['odometry']['child_frame'])] = cfg['name']
            if 'calibration' in cfg:
                for tf in cfg['calibration']['transform_params']:
                    tfs_to_odom[(tf['from_frame'], tf['to_frame'])] = cfg['name']
            if 'bag_static_tfs' in cfg:
                for tf in cfg['bag_static_tfs']:
                    tfs_to_odom[(tf['from_frame'], tf['to_frame'])] = cfg['name']

        for inv_tf in tf_inversions:
            tfs_to_invert.add((inv_tf['from_frame'], inv_tf['to_frame']))

        bag_fps = sorted([x for x in os.listdir(rosbag_fp) if '.mcap' in x])

        # have every frame keep track of tf to its parent
        # separate the frames dicts by tree, where trees are both vehicle and odometry-specific trees
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

                    # invert transforms that match one of the from_frame, to_frame pairs in tfs_to_invert
                    need_invert = (src_frame, dst_frame) in tfs_to_invert

                    if need_invert:
                        src_frame = tf_msg.child_frame_id
                        dst_frame = tf_msg.header.frame_id

                    # based on the src_frame and dst_frame, determine which tree this tf belongs to
                    tree_name = tfs_to_odom[(src_frame, dst_frame)] if (src_frame, dst_frame) in tfs_to_odom else 'vehicle'

                    if tree_name not in frames.keys():
                        frames[tree_name] = {}

                    if dst_frame not in frames[tree_name].keys():
                        frames[tree_name][dst_frame] = {
                            'frame_id': dst_frame,
                            'parent_frame_id': src_frame,
                            'is_static': topic == '/tf_static',
                            'transforms': np.zeros([0, 7]),
                            'times': np.zeros(0)
                        }
                    else:
                        # Skip transforms that try to rewire the tf tree
                        if src_frame != frames[tree_name][dst_frame]['parent_frame_id']:
                            if (src_frame, dst_frame) not in tf_rewire_warnings:
                                print(f"Warning: Skipping transform {src_frame}->{dst_frame} in {tree_name} tree (already have {frames[tree_name][dst_frame]['parent_frame_id']}->{dst_frame})")
                                tf_rewire_warnings.add((src_frame, dst_frame))
                            continue

                    if dt > 0. and len(frames[tree_name][dst_frame]['times']) > 0:
                        if t - frames[tree_name][dst_frame]['times'][-1] < dt:
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

                    if need_invert:
                        transform_htm = pose_to_htm(tf_data)
                        transform_htm = np.linalg.inv(transform_htm)
                        tf_data = htm_to_pose(transform_htm)

                    frames[tree_name][dst_frame]['times'] = np.append(frames[tree_name][dst_frame]['times'], t)
                    frames[tree_name][dst_frame]['transforms'] = np.append(frames[tree_name][dst_frame]['transforms'], tf_data.reshape(1, 7), axis=0)
                cnt += 1
        # postproc to check for any non-static transforms in the vehicle tree as usually the vehicle tree just has static tfs
        for nodes in frames['vehicle'].values():
            if not nodes['is_static']:
                print(f"""Warning: Non-static transform {nodes['parent_frame_id']}->{nodes['frame_id']} found in vehicle tree, 
                        double check that this transform is not an odometry-related transform that should be specified 
                        in the odometry config""")
        # build vehicle tf tree
        tf_manager.vehicle_tree = TfTree(nodes=[TfNode(**v) for v in frames['vehicle'].values()])
        # Before creating odom trees, normalize the odom tf to be describing the vehicle frame
        # use vehicle tree to lookup the transform between the odom tf's child frame and the vehicle frame
        odom_frames = {k: v for k, v in frames.items() if k != 'vehicle'}
        TfManager.normalize_odom_tfs(odom_frames, odometry_configs, tf_manager)
        # build odom-specific tf trees 
        odom_nodes = {name: [TfNode(**v) for v in nodes.values()] for name, nodes in odom_frames.items() if name != 'vehicle'}
        tf_manager.odom_trees = {name: TfTree(nodes=nodes) for name, nodes in odom_nodes.items()}

        if odometry_configs:
            print("Building connections to the vehicle tf tree for odometry sources")
            tf_manager.build_bridges(odometry_configs)

        return tf_manager

    def get_valid_times(self, frame1, frame2):
        """
        Get the range of times that we can transform between frame1 and frame2
        """
        if frame1 == frame2:
            return -float('inf'), float('inf')

        tree = self.get_lookup_tree(frame1, frame2)
        lca_paths = tree.get_lca_paths(frame1, frame2)
        if lca_paths:
            all_tfs = lca_paths[0] + lca_paths[1]
            tmin = max([node.t_min for node in all_tfs]) if all_tfs else -float('inf')
            tmax = min([node.t_max for node in all_tfs]) if all_tfs else float('inf')

            return tmin, tmax
        else:
            return float('inf'), -float('inf')

    def get_valid_times_from_list(self, frame_list):
        """
        Get valid sample times for a list of frames.
        With multiple odometry sources, there isn't one global tree, so each
        odometry source + vehicle tree group should be checked separately
        """
        tmin = -float('inf')
        tmax = float('inf')

        frame_list = [x for x in frame_list if self.has_frame(x)]

        odom_groups = {}
        vehicle_frames = []

        # assign frames to either an odometry source or vehicle tree
        for f in frame_list:
            odoms = [x for x in self.find_tree_for_frame(f) if x != 'vehicle']
            if odoms:
                for name in odoms:
                    odom_groups.setdefault(name, []).append(f)
            else:
                vehicle_frames.append(f)
        
        if not odom_groups:
            # all frames in frame_list are in the vehicle tree
            if len(vehicle_frames) < 2:
                return tmin, tmax

            for frame in vehicle_frames:
                _tmin, _tmax = self.get_valid_times(frame, vehicle_frames[0])
                tmin = max(tmin, _tmin)
                tmax = min(tmax, _tmax)

            return tmin, tmax

        # check times against each group of odometry source + vehicle frames separately
        for name, odom_frames in odom_groups.items():
            frame_group = odom_frames + vehicle_frames

            for frame in frame_group:
                _tmin, _tmax = self.get_valid_times(frame, odom_frames[0])
                tmin = max(tmin, _tmin)
                tmax = min(tmax, _tmax)

        return tmin, tmax

    def can_transform(self, src_frame, dst_frame, t):
        tmin, tmax = self.get_valid_times(src_frame, dst_frame)
        return t >= tmin and t <= tmax
    
    def get_transform_from_tree(self, frame1, frame2, t, tree):
        """
        Get the transform from frame1 to frame2 at time t, given the tree to lookup from
        """
        frame1_path, frame2_path = tree.get_lca_paths(frame1, frame2)

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

    def get_transform(self, frame1, frame2, t):
        """
        Get the transform from frame1 to frame2 at time t
        """
        tree = self.get_lookup_tree(frame1, frame2)
        tf = self.get_transform_from_tree(frame1, frame2, t, tree)

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