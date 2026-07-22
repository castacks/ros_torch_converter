import os
import numpy as np

import warnings

from tartandriver_utils.os_utils import load_yaml, save_yaml

def update_info_file(base_dir, key, value):
    info_fp = os.path.join(base_dir, 'info.yaml')

    if os.path.exists(info_fp):
        info_dict = load_yaml(info_fp) or {}  # guard against empty file during concurrent write

        if key in info_dict and info_dict[key] == value:
            return  # already correct — no write needed (handles parallel workers reading pre-written files)

        if key in info_dict and info_dict[key] != value:
            warnings.simplefilter('once')
            warnings.warn(f'for {base_dir}:{key}, {value} doesnt match previous value {info_dict[key]}! Check that this is correct!')

        info_dict[key] = value

    else:
        info_dict = {key: value}

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

def update_timestamp_file(base_dir, idx, stamp, file='timestamps.txt'):
    timestamp_fp = os.path.join(base_dir, file)

    if os.path.exists(timestamp_fp):
        data = np.loadtxt(timestamp_fp)
        if data.size == 0:  # guard against empty file during concurrent write
            timestamps = -np.ones(idx + 1)
        else:
            timestamps = data.reshape(-1)
            if idx < timestamps.shape[0] and timestamps[idx] == stamp:
                return  # already correct — no write needed (handles parallel workers reading pre-written files)
            if idx >= timestamps.shape[0]:
                temp_timestamps = -np.ones(idx + 1)
                temp_timestamps[:timestamps.shape[0]] = timestamps
                timestamps = temp_timestamps
    else:
        timestamps = -np.ones(idx + 1)

    timestamps[idx] = stamp
    with open(timestamp_fp, 'w') as f:
        np.savetxt(f, timestamps)

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