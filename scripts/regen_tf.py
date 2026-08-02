#!/usr/bin/env python3
"""
Regenerate the tf/ output for already-extracted KITTI sequences from a corrected calib file.

The extractor builds its tf tree ONLY from the calib file (ros2bag_2_kitti.py uses an empty
TfManager, not the bag), so tf/ can be regenerated offline with no bag reads.

For each sequence dir (identified by containing target_timestamps.txt), the existing tf/ dir is
moved to tf_prev/ (once, as a backup) and a fresh tf/ is written from the calib.

Usage:
  python3 regen_tf.py --root <extracted_root> --calib_file <handheld.yaml> [--dry] [--seq <one_dir>]
"""
import os
import shutil
import argparse

from ros_torch_converter.tf_manager import TfManager
import yaml


def find_sequence_dirs(root):
    seqs = []
    for r, d, f in os.walk(root):
        if 'target_timestamps.txt' in f:
            seqs.append(r)
    return sorted(seqs)


def regen_one(seq_dir, calib_config, dry=False, backup=True):
    tf_dir = os.path.join(seq_dir, 'tf')
    prev_dir = os.path.join(seq_dir, 'tf_prev')

    if dry:
        print('[DRY] would regenerate tf for', seq_dir)
        return

    if os.path.isdir(tf_dir) and backup and not os.path.isdir(prev_dir):
        shutil.move(tf_dir, prev_dir)
    elif os.path.isdir(tf_dir):
        # tf_prev already exists (already backed up once) -> just drop the stale tf
        shutil.rmtree(tf_dir)

    os.makedirs(tf_dir, exist_ok=True)

    tfm = TfManager(device='cpu')
    tfm.update_from_calib_config(calib_config)
    tfm.to_kitti(seq_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--calib_file', type=str, required=True)
    ap.add_argument('--seq', type=str, default=None, help='regenerate a single sequence dir only')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--no_backup', action='store_true')
    args = ap.parse_args()

    calib_config = yaml.safe_load(open(args.calib_file, 'r'))

    seqs = [args.seq] if args.seq else find_sequence_dirs(args.root)
    print('found {} sequence(s)'.format(len(seqs)))

    for i, s in enumerate(seqs):
        print('({}/{}) {}'.format(i + 1, len(seqs), s))
        regen_one(s, calib_config, dry=args.dry, backup=not args.no_backup)

    print('done.')
