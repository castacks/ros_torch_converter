import os
import re
import yaml
import numpy as np

import warnings

from pathlib import Path

from tartandriver_utils.os_utils import load_yaml, save_yaml

def ros1_bag_topics(path):
    """
    Cheaply read JUST the set of topics in a ROS1 .bag, without building its message index.

    A ROS1 v2.0 bag stores all of its connection records (which carry the topic names) in an
    index region at the very end of the file. rosbags' Reader.open() reads those connection
    records *before* it does the expensive work of seeking to every chunk and building the
    per-message index. We replicate only that cheap prefix here: read the bag header to get
    index_pos / conn_count, seek to index_pos, and read conn_count connection records (reusing
    the upstream Reader.read_connection parser). No chunk or message data is touched, so peeking
    an irrelevant bag costs a header read + one seek + a few small records instead of a full
    index build over the whole (possibly multi-GB) bag.

    Returns the set of topic strings, or None if the topics could not be determined (caller
    should then keep the bag rather than risk dropping a needed one).
    """
    from rosbags.rosbag1.reader import Reader, Header, RecordType
    try:
        reader = Reader(path)
        reader.bio = reader.path.open('rb')
        try:
            magic = reader.bio.readline().decode()
            if not re.match(r'#ROSBAG V2.0\n', magic):
                return None  # not a v2.0 bag we know how to peek; let the caller handle it
            header = Header.read(reader.bio, RecordType.BAGHEADER)
            index_pos = header.get_uint64('index_pos')
            conn_count = header.get_uint32('conn_count')
            if index_pos == 0 or conn_count == 0:
                return None  # unindexed / empty: can't peek cheaply, keep the bag
            reader.bio.seek(index_pos)
            return {reader.read_connection().topic for _ in range(conn_count)}
        finally:
            reader.bio.close()
    except Exception as e:
        print('topic peek failed for {} ({}); keeping it'.format(path, e))
        return None

def select_ros1_bags(src_dir, wanted_topics, verbose=True):
    """
    Return the sorted list of .bag Paths in src_dir that carry at least one topic in
    wanted_topics, using the cheap ros1_bag_topics() peek (no full index build).

    This lets callers point at a whole bag folder while only opening/indexing the bags that
    actually contain a topic of interest. Bags whose topics cannot be peeked (peek -> None)
    are kept, so a needed bag is never silently dropped.
    """
    wanted = set(wanted_topics)
    bag_fps = sorted(x for x in os.listdir(src_dir) if x.endswith('.bag'))
    kept, skipped = [], []
    for bfp in bag_fps:
        topics = ros1_bag_topics(Path(src_dir) / bfp)
        (kept if (topics is None or (topics & wanted)) else skipped).append(bfp)
    if verbose and skipped:
        print('skipping {} bag(s) with none of {} (topic peek, not indexed):'.format(
            len(skipped), sorted(wanted)))
        for bfp in skipped:
            print('\t' + bfp)
    return [Path(src_dir) / x for x in kept]

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