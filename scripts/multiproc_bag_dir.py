import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='base dir of run dirs to proc')
    parser.add_argument('--dst_dir', type=str, required=True, help='base dir of save')
    parser.add_argument('--config_fp', type=str, required=True, help='path to save config')
    args = parser.parse_args()

    rdirs = os.listdir(args.src_dir)
    rdirs = reversed(sorted(rdirs))

    base_cmd = "python3 ros2bag_2_kitti.py --config {} --dst_dir {} --src_dir {}"

    cmds = []

    for rdir in rdirs:
        src_dir = os.path.join(args.src_dir, rdir)
        dst_dir = os.path.join(args.dst_dir, rdir)
        cmd = base_cmd.format(args.config_fp, dst_dir, src_dir)
        cmds.append(cmd)

    out = " && ".join(cmds)

    print(out)
