#!/usr/bin/env python3
"""
Backfill full-rate topics (IMU) into already-extracted KITTI sequences, without re-running the
full image extraction. For each extracted sequence dir (contains target_timestamps.txt), the
matching rosbag dir is located by mirroring the relative path under --bags_root, and every
message of each `full_rate_topics` entry is dumped into the sequence dir.

Usage:
  python3 backfill_imu.py --ext_root <extracted_root> --bags_root <bags_root> \
      --config <tartan_rgbt.yaml> [--overwrite] [--seq <one_ext_dir>]
"""
import os
import sys
import argparse

import yaml
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ros2bag_2_kitti import extract_full_rate_topics  # noqa: E402


def find_sequence_dirs(root):
    seqs = []
    for r, d, f in os.walk(root):
        if 'target_timestamps.txt' in f:
            seqs.append(r)
    return sorted(seqs)


def bag_dir_for(ext_dir, ext_root, bags_root):
    rel = os.path.relpath(ext_dir, ext_root)
    return os.path.join(bags_root, rel)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ext_root', type=str, required=True)
    ap.add_argument('--bags_root', type=str, required=True)
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--seq', type=str, default=None)
    ap.add_argument('--overwrite', action='store_true',
                    help='re-extract even if the output folder already exists')
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    frt = config.get('full_rate_topics', [])
    assert frt, 'config has no full_rate_topics'
    out_names = [x['name'] for x in frt]

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    seqs = [args.seq] if args.seq else find_sequence_dirs(args.ext_root)
    print('found {} sequence(s); full_rate outputs: {}'.format(len(seqs), out_names))

    done, skipped, missing = 0, 0, 0
    for i, ext_dir in enumerate(seqs):
        bag_dir = bag_dir_for(ext_dir, args.ext_root, args.bags_root)
        has_all = all(os.path.isdir(os.path.join(ext_dir, n)) for n in out_names)
        if has_all and not args.overwrite:
            print('({}/{}) SKIP (exists) {}'.format(i + 1, len(seqs), ext_dir))
            skipped += 1
            continue
        if not (os.path.isdir(bag_dir) and any(f.endswith('.mcap') for f in os.listdir(bag_dir))):
            print('({}/{}) MISSING BAG {} -> {}'.format(i + 1, len(seqs), ext_dir, bag_dir))
            missing += 1
            continue

        print('({}/{}) {}  <-  {}'.format(i + 1, len(seqs), ext_dir, bag_dir))
        extract_full_rate_topics(Path(bag_dir), typestore, config, ext_dir)
        done += 1

    print('done. extracted={} skipped={} missing_bag={}'.format(done, skipped, missing))
