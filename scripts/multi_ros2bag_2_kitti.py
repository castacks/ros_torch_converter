import os
import tqdm
import argparse
import subprocess

from tartandriver_utils.os_utils import is_rosbag_dir

if __name__ == '__main__':
    """
    recommend running this script as python3 multi_rosbag_to_kitti.py 2>&1 | tee log.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='base dir of run dirs to proc')
    parser.add_argument('--dst_dir', type=str, required=True, help='base dir of save')
    parser.add_argument('--config_fp', type=str, required=True, help='path to save config')
    args = parser.parse_args()

    rosbag_dirs = []

    for root, dirs, files in os.walk(args.src_dir):
        if is_rosbag_dir(root):
            rosbag_dirs.append(root)

    print('found the following rosbag dirs ({} total):'.format(len(rosbag_dirs)))
    for rdir in sorted(rosbag_dirs):
        print('\t' + rdir)

    base_cmd = "python3 ros2bag_2_kitti.py --config {} --src_dir {} --dst_dir {} --dryrun --no_plot --force"
    # base_cmd = "ls {}" #if theres a bug in the second part of the script this serves as a cache

    success_rosbag_dirs = []
    success_dirs = []
    fail_dirs = []

    for ri, rosbag_dir in tqdm.tqdm(enumerate(rosbag_dirs)):
        print("Proc {} ({}/{})".format(rosbag_dir, ri+1, len(rosbag_dirs)))
        relpath = os.path.relpath(rosbag_dir, start=args.src_dir)
        dst_path = os.path.join(args.dst_dir, relpath)

        cmd = base_cmd.format(args.config_fp, rosbag_dir, dst_path)
        # cmd = base_cmd.format(dst_path)

        res = subprocess.run(cmd.split(" "))

        if res.returncode == 0:
            success_dirs.append(dst_path)
            success_rosbag_dirs.append(rosbag_dir)
        else:
            fail_dirs.append(dst_path)

    print('can extract data for {}/{} rosbag dirs'.format(len(success_dirs), len(rosbag_dirs)))

    base_cmd = "python3 ros2bag_2_kitti.py --config {} --src_dir {} --dst_dir {} --force --no_plot"

    success_proc_dirs = []
    fail_proc_dirs = []

    for ri, (rosbag_dir, dst_dir) in tqdm.tqdm(enumerate(zip(success_rosbag_dirs, success_dirs))):
        print("Proc {} ({}/{})".format(rosbag_dir, ri+1, len(success_dirs)))
        cmd = base_cmd.format(args.config_fp, rosbag_dir, dst_dir)

        res = subprocess.run(cmd.split(" "))

        if res.returncode == 0:
            success_proc_dirs.append(dst_path)
        else:
            fail_proc_dirs.append(dst_path)

    print('successfully extracted data for {}/{} proc dirs'.format(len(success_proc_dirs), len(success_dirs)))

    if len(fail_proc_dirs) > 0:
        print('should manually check:')
        for fp in fail_proc_dirs:
            print('    ' + fp)