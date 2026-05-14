
import os
import yaml
import numpy as np

import warnings

from tartandriver_utils.os_utils import load_yaml, save_yaml


def update_frame_file(base_dir, idx, frame_type, frame_id):
    frame_fp = os.path.join(base_dir, 'frames.yaml')

    if os.path.exists(frame_fp):
        with open(frame_fp, 'r') as fp:
            frame_dict = yaml.safe_load(fp)

        if frame_type in frame_dict.keys() and frame_dict[frame_type] != frame_id:
            warnings.simplefilter('once')
            warnings.warn(f'frame id {frame_type} doesnt match previous value {frame_dict[frame_type]}! Check that your frames are correct!')

        frame_dict[frame_type] = frame_id

    else:
        frame_dict = {frame_type:frame_id}

    with open(frame_fp, 'w') as fp:
        yaml.dump(frame_dict, fp)

def read_frame_file(base_dir, idx, frame_type, allow_missing=True):
    frame_fp = os.path.join(base_dir, 'frames.yaml')

    if os.path.exists(frame_fp):
        with open(frame_fp, 'r') as fp:
            frame_dict = yaml.safe_load(fp)

        return frame_dict[frame_type]
    else:
        if allow_missing:
            return "NULL"
        else:
            raise Exception("Couldnt find frame file and allow_missing=False")

def update_info_file(base_dir, key, value):
    info_fp = os.path.join(base_dir, 'info.yaml')

    if os.path.exists(info_fp):
        info_dict = load_yaml(info_fp)

        if key in info_dict.keys() and info_dict[key] != value:
            warnings.simplefilter('once')
            warnings.warn(f'for {base_dir}:{key}, {value} doesnt match previous value {info_dict[key]}! Check that this is correct!')

        info_dict[key] = value

    else:
        info_dict = {key:value}

    save_yaml(info_dict, info_fp)    

def read_info_file(base_dir, key, allow_missing=True, default_value="NULL"):
    info_fp = os.path.join(base_dir, 'info.yaml')

    if os.path.exists(info_fp):
        info_dict = load_yaml(info_fp)

        if allow_missing:
            return info_dict.get(key, default_value)
        else:
            return info_dict[key]
    else:
        if allow_missing:
            return default_value
        else:
            raise Exception("Couldnt find frame file and allow_missing=False")


def update_timestamp_file(base_dir, idx, stamp):
    timestamp_fp = os.path.join(base_dir, 'timestamps.txt')

    if os.path.exists(timestamp_fp):
        timestamps = np.loadtxt(timestamp_fp).reshape(-1)
        if idx >= timestamps.shape[0]:
            temp_timestamps = -np.ones(idx+1)
            temp_timestamps[:timestamps.shape[0]] = timestamps
            temp_timestamps[idx] = stamp
            timestamps = temp_timestamps
        else:
            timestamps[idx] = stamp
    else:
        timestamps = -np.ones(idx+1)
        timestamps[idx] = stamp

    np.savetxt(timestamp_fp, timestamps)

# new
# def update_timestamp_file(base_dir, idx, stamp, file='timestamps.txt'):
#     timestamp_fp = os.path.join(base_dir, file)

#     if os.path.exists(timestamp_fp):
#         timestamps = np.loadtxt(timestamp_fp).reshape(-1)
#         if idx >= timestamps.shape[0]:
#             temp_timestamps = -np.ones(idx+1)
#             temp_timestamps[:timestamps.shape[0]] = timestamps
#             temp_timestamps[idx] = stamp
#             timestamps = temp_timestamps
#     else:
#         timestamps = -np.ones(idx+1)

#     timestamps[idx] = stamp

#     np.savetxt(timestamp_fp, timestamps)

def read_timestamp_file(base_dir, idx, allow_missing=True, file='timestamps.txt'):
    timestamp_fp = os.path.join(base_dir, file)
    if not os.path.exists(timestamp_fp):
        if allow_missing:
            return -1.
        else:
            raise Exception("Couldnt find timestamp file and allow_missing=False")

    else:
        timestamps = np.loadtxt(timestamp_fp).reshape(-1)

        return timestamps[idx]

# OLD

# import os
# import yaml
# import numpy as np

# import warnings

# def update_frame_file(base_dir, idx, frame_type, frame_id):
#     frame_fp = os.path.join(base_dir, 'frames.yaml')

#     if os.path.exists(frame_fp):
#         with open(frame_fp, 'r') as fp:
#             frame_dict = yaml.safe_load(fp)

#         if frame_type in frame_dict.keys() and frame_dict[frame_type] != frame_id:
#             warnings.simplefilter('once')
#             warnings.warn(f'frame id {frame_type} doesnt match previous value {frame_dict[frame_type]}! Check that your frames are correct!')

#         frame_dict[frame_type] = frame_id

#     else:
#         frame_dict = {frame_type:frame_id}

#     with open(frame_fp, 'w') as fp:
#         yaml.dump(frame_dict, fp)

# def read_frame_file(base_dir, idx, frame_type, allow_missing=True):
#     frame_fp = os.path.join(base_dir, 'frames.yaml')

#     if os.path.exists(frame_fp):
#         with open(frame_fp, 'r') as fp:
#             frame_dict = yaml.safe_load(fp)

#         return frame_dict[frame_type]
#     else:
#         if allow_missing:
#             return "NULL"
#         else:
#             raise Exception("Couldnt find frame file and allow_missing=False")

# def update_timestamp_file(base_dir, idx, stamp):
#     timestamp_fp = os.path.join(base_dir, 'timestamps.txt')

#     if os.path.exists(timestamp_fp):
#         timestamps = np.loadtxt(timestamp_fp).reshape(-1)
#         if idx >= timestamps.shape[0]:
#             temp_timestamps = -np.ones(idx+1)
#             temp_timestamps[:timestamps.shape[0]] = timestamps
#             temp_timestamps[idx] = stamp
#             timestamps = temp_timestamps
#         else:
#             timestamps[idx] = stamp
#     else:
#         timestamps = -np.ones(idx+1)
#         timestamps[idx] = stamp

#     np.savetxt(timestamp_fp, timestamps)

# def read_timestamp_file(base_dir, idx, allow_missing=True):
#     timestamp_fp = os.path.join(base_dir, 'timestamps.txt')
#     if not os.path.exists(timestamp_fp):
#         if allow_missing:
#             return -1.
#         else:
#             raise Exception("Couldnt find timestamp file and allow_missing=False")

#     else:
#         timestamps = np.loadtxt(timestamp_fp).reshape(-1)

#         return timestamps[idx]
    