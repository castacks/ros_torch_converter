'''Compare ROS-extracted processed thermal PNGs to get_thermal_data.py output.

ROS ThermalImageTorch frames are 3-channel stacked gray; the script writes
mono8. Both are reduced to one channel before absdiff.

Usage (KITTI run dir, --dataset for get_thermal_data is <kitti>/sensors):

  python3 scripts/compare_thermal_kitti.py --kitti_dir /path/to/run
'''

import argparse
import json
import os
import sys

import cv2
import numpy as np


def to_gray(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[..., 0]
    raise ValueError(f"unexpected image shape {img.shape}")


def png_names(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(n for n in os.listdir(directory) if n.endswith('.png'))


def compare_side(ros_dir, script_dir):
    ros_names = png_names(ros_dir)
    script_names = set(png_names(script_dir))
    missing_script = [n for n in ros_names if n not in script_names]
    extra_script = sorted(script_names - set(ros_names))

    n_diff_pixels = 0
    max_abs = 0
    hist = np.zeros(256, dtype=np.int64)
    worst_frame = None
    worst_max = -1
    n_compared = 0

    for name in ros_names:
        if name not in script_names:
            continue
        ros = cv2.imread(os.path.join(ros_dir, name), cv2.IMREAD_UNCHANGED)
        scr = cv2.imread(os.path.join(script_dir, name), cv2.IMREAD_UNCHANGED)
        if ros is None or scr is None:
            continue
        ros_g = to_gray(ros).astype(np.int16)
        scr_g = to_gray(scr).astype(np.int16)
        if ros_g.shape != scr_g.shape:
            raise ValueError(f"{name}: shape {ros_g.shape} vs {scr_g.shape}")
        diff = np.abs(ros_g - scr_g)
        frame_max = int(diff.max()) if diff.size else 0
        frame_n = int((diff > 0).sum())
        n_diff_pixels += frame_n
        max_abs = max(max_abs, frame_max)
        hist += np.bincount(diff.ravel(), minlength=256)[:256]
        if frame_max > worst_max:
            worst_max = frame_max
            worst_frame = name
        n_compared += 1

    return {
        'ros_dir': ros_dir,
        'script_dir': script_dir,
        'n_frames_ros': len(ros_names),
        'n_frames_script': len(script_names),
        'n_compared': n_compared,
        'n_diff_pixels': int(n_diff_pixels),
        'max_abs': int(max_abs),
        'hist': hist.tolist(),
        'worst_frame': worst_frame,
        'missing_script': missing_script,
        'extra_script': extra_script,
    }


def print_opencv_flags():
    print(f"[thermal_verify] cv2={cv2.__version__} file={cv2.__file__}")
    print(f"[thermal_verify] ocl.have={cv2.ocl.haveOpenCL()} ocl.use={cv2.ocl.useOpenCL()}")
    print(f"[thermal_verify] AVX2={cv2.checkHardwareSupport(cv2.CPU_AVX2)} "
          f"AVX512_SKX={cv2.checkHardwareSupport(cv2.CPU_AVX512_SKX)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', type=str, required=True,
                        help='KITTI run root (contains sensors/)')
    parser.add_argument('--group', type=str, default='sensors')
    parser.add_argument('--ros_left', type=str, default='thermal_left_processed')
    parser.add_argument('--ros_right', type=str, default='thermal_right_processed')
    parser.add_argument('--script_left', type=str, default='thermal_left_processed_script')
    parser.add_argument('--script_right', type=str, default='thermal_right_processed_script')
    parser.add_argument('--out', type=str, default=None,
                        help='report path (default: <kitti_dir>/thermal_verify_report.json)')
    args = parser.parse_args()

    print_opencv_flags()

    group_dir = os.path.join(args.kitti_dir, args.group)
    report = {
        'kitti_dir': args.kitti_dir,
        'left': compare_side(
            os.path.join(group_dir, args.ros_left),
            os.path.join(group_dir, args.script_left),
        ),
        'right': compare_side(
            os.path.join(group_dir, args.ros_right),
            os.path.join(group_dir, args.script_right),
        ),
    }

    left, right = report['left'], report['right']
    print(f"[thermal_verify] left max={left['max_abs']} n={left['n_diff_pixels']} "
          f"frames={left['n_compared']} | right max={right['max_abs']} "
          f"n={right['n_diff_pixels']} frames={right['n_compared']}")

    out_path = args.out or os.path.join(args.kitti_dir, 'thermal_verify_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[thermal_verify] wrote {out_path}")

    if left['n_compared'] == 0 and right['n_compared'] == 0:
        print('[thermal_verify] ERROR: no overlapping frames to compare', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
