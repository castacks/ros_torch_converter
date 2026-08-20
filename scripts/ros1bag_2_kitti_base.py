#!/usr/bin/env python3
import os
import yaml
import rosbag
import rospy
import numpy as np
from scipy.interpolate import interp1d

# For handling TF trees manually from a bag
import tf2_msgs.msg
import geometry_msgs.msg

class BagDatasetExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.dt = self.config['dt']
        self.interp_tol = self.config['interp_tol']
        self.topics_cfg = self.config['topics']
        
        # Map topic names to their configuration
        self.topic_map = {t['topic']: t for t in self.topics_cfg}
        self.topic_list = list(self.topic_map.keys())

    def extract(self, bag_path):
        print(f"Opening bag: {bag_path}")
        bag = rosbag.Bag(bag_path)
        
        # Initialize data storage
        # { topic_short_name: { 'times': [], 'data': [] } }
        raw_data = {t['name']: {'times': [], 'data': []} for t in self.topics_cfg}
        tf_data = [] # Store raw TF messages

        # 1. Read all requested topics and TFs from the bag
        topics_to_read = self.topic_list + ['/tf', '/tf_static']
        
        start_time = None
        end_time = None

        for topic, msg, t in bag.read_messages(topics=topics_to_read):
            timestamp = t.to_sec()
            if start_time is None:
                start_time = timestamp
            end_time = timestamp

            if topic in ['/tf', '/tf_static']:
                tf_data.append((timestamp, msg))
            else:
                cfg = self.topic_map[topic]
                short_name = cfg['name']
                raw_data[short_name]['times'].append(timestamp)
                # Store the raw message object (or custom extract logic based on type)
                raw_data[short_name]['data'].append(msg)

        bag.close()
        
        if not start_time or not end_time:
            print("No data found.")
            return

        print(f"Bag time range: {start_time:.2f} to {end_time:.2f} ({end_time - start_time:.2f}s)")

        # 2. Create the master synchronous time grid
        # Padding slightly inwards to avoid edge extrapolation artifacts
        grid_start = start_time + self.dt
        grid_end = end_time - self.dt
        master_grid = np.arange(grid_start, grid_end, self.dt)
        num_samples = len(master_grid)
        print(f"Generated {num_samples} dataset steps with dt={self.dt}")

        # 3. Synchronize / Resample data to the grid
        synced_dataset = {
            'timestamps': master_grid,
            'data': {t['name']: [None] * num_samples for t in self.topics_cfg}
        }

        for cfg in self.topics_cfg:
            name = cfg['name']
            msg_type = cfg['type']
            times = np.array(raw_data[name]['times'])
            msgs = raw_data[name]['data']
            
            if len(times) == 0:
                print(f"Warning: No messages found for topic {cfg['topic']}")
                continue

            print(f"Syncing {name} ({msg_type})...")

            # Handle numeric vs discrete/heavy data types
            if msg_type in ['OdomRBState', 'Float32']: 
                # Continuous numeric data -> Linear Interpolation
                synced_dataset['data'][name] = self._interpolate_numeric(master_grid, times, msgs, msg_type)
            else:
                # Heavy data (Image, PointCloud, Intrinsics) -> Nearest Neighbor within tolerance
                synced_dataset['data'][name] = self._nearest_neighbor_sync(master_grid, times, msgs)

        # 4. Synchronize TFs to the master grid
        print("Syncing TF tree states...")
        synced_dataset['tf'] = self._sync_tfs(master_grid, tf_data)

        return synced_dataset

    def _nearest_neighbor_sync(self, grid, times, msgs):
        """Finds the closest message in time for each grid step within interp_tol."""
        synced_msgs = [None] * len(grid)
        for i, target_t in enumerate(grid):
            idx = np.searchsorted(times, target_t)
            
            # Check bounding indices to find the closest one
            indices_to_check = [idx - 1, idx, idx + 1]
            best_idx = None
            min_dt = self.interp_tol
            
            for j in indices_to_check:
                if 0 <= j < len(times):
                    diff = abs(times[j] - target_t)
                    if diff < min_dt:
                        min_dt = diff
                        best_idx = j
            
            if best_idx is not None:
                synced_msgs[i] = msgs[best_idx]
        return synced_msgs

    def _interpolate_numeric(self, grid, times, msgs, msg_type):
        """Linearly interpolates numerical structures like Odometry or Floats."""
        synced_msgs = [None] * len(grid)
        
        if msg_type == 'Float32':
            values = np.array([m.data for m in msgs])
            # Interpolate, fill values outside bounds with NaN
            f = interp1d(times, values, kind='linear', bounds_error=False, fill_value=np.nan)
            grid_vals = f(grid)
            
            # Build back into ROS messages if desired, or store raw
            import std_msgs.msg
            for i, val in enumerate(grid_vals):
                if not np.isnan(val):
                    synced_msgs[i] = std_msgs.msg.Float32(data=float(val))
                    
        elif msg_type == 'OdomRBState' or 'nav_msgs/Odometry' in msg_type:
            # Assuming standard nav_msgs/Odometry or structure containing pose/twist
            # For this example, we extract positions/orientations arrays
            # (If OdomRBState is custom, adjust attribute names accordingly)
            x = np.array([m.pose.pose.position.x for m in msgs])
            y = np.array([m.pose.pose.position.y for m in msgs])
            z = np.array([m.pose.pose.position.z for m in msgs])
            
            fx = interp1d(times, x, bounds_error=False, fill_value=np.nan)
            fy = interp1d(times, y, bounds_error=False, fill_value=np.nan)
            fz = interp1d(times, z, bounds_error=False, fill_value=np.nan)
            
            # Note: For strict 3D rotations, Slerp should ideally be used for quaternions. 
            # Doing basic linear approximation here for brevity.
            qx = np.array([m.pose.pose.orientation.x for m in msgs])
            qy = np.array([m.pose.pose.orientation.y for m in msgs])
            qz = np.array([m.pose.pose.orientation.z for m in msgs])
            qw = np.array([m.pose.pose.orientation.w for m in msgs])
            
            f_qx = interp1d(times, qx, bounds_error=False)
            f_qy = interp1d(times, qy, bounds_error=False)
            f_qz = interp1d(times, qz, bounds_error=False)
            f_qw = interp1d(times, qw, bounds_error=False)

            import nav_msgs.msg
            for i, t in enumerate(grid):
                # Ensure we are within the interpolation tolerance limit
                closest_idx = np.searchsorted(times, t)
                if closest_idx == 0 or closest_idx >= len(times): continue
                if abs(times[closest_idx] - t) > self.interp_tol and abs(times[closest_idx-1] - t) > self.interp_tol:
                    continue
                
                # Reconstruct an odom message frame
                out_msg = nav_msgs.msg.Odometry()
                out_msg.pose.pose.position.x = float(fx(t))
                out_msg.pose.pose.position.y = float(fy(t))
                out_msg.pose.pose.position.z = float(fz(t))
                
                # Normalize interpolated quaternion
                q = np.array([f_qx(t), f_qy(t), f_qz(t), f_qw(t)])
                q /= np.linalg.norm(q)
                out_msg.pose.pose.orientation.x = float(q[0])
                out_msg.pose.pose.orientation.y = float(q[1])
                out_msg.pose.pose.orientation.z = float(q[2])
                out_msg.pose.pose.orientation.w = float(q[3])
                synced_msgs[i] = out_msg
                
        return synced_msgs

    def _sync_tfs(self, grid, tf_data):
        """
        Builds a TF snapshot at every step on the grid using a look-behind approach.
        Stores a mapping of (parent_frame, child_frame) -> geometry_msgs/Transform
        """
        synced_tfs = [{} for _ in range(len(grid))]
        
        # Sort tf data by time stamp just in case
        tf_data.sort(key=lambda x: x[0])
        
        tf_idx = 0
        current_tf_tree = {} # (parent, child) -> transform_msg

        for i, target_t in enumerate(grid):
            # Advance through TF messages until we surpass the target grid time
            while tf_idx < len(tf_data) and tf_data[tf_idx][0] <= target_t:
                _, msg = tf_data[tf_idx]
                for transform in msg.transforms:
                    key = (transform.header.frame_id, transform.child_frame_id)
                    current_tf_tree[key] = transform
                tf_idx += 1
            
            # Save a snapshot copy of the TF tree state at this specific grid time step
            synced_tfs[i] = dict(current_tf_tree)
            
        return synced_tfs

# --- Example Usage Execution ---
if __name__ == '__main__':
    # Save your yaml snippet as 'config.yaml'
    config_file = "config.yaml"
    bag_file = "your_data.bag"
    
    # Quick creation of dummy config for testing if file doesn't exist
    if not os.path.exists(config_file):
        print(f"Please create the config file: {config_file}")
        exit(1)

    extractor = BagDatasetExtractor(config_file)
    dataset = extractor.extract(bag_file)
    
    if dataset:
        # Example on how to look at sample index `n`
        n = 100
        print(f"\n--- Checking Data Point n={n} ---")
        print(f"Timestamp: {dataset['timestamps'][n]}")
        for name in dataset['data'].keys():
            msg = dataset['data'][name][n]
            print(f"Modality '{name}': {'Data Present' if msg is not None else 'MISSING (out of tolerance)'}")
            
        print(f"Number of TF pairs tracked at step {n}: {len(dataset['tf'][n])}")