#!/bin/bash
# Standalone TartanRGBT extraction runner — uses ONLY tartan_rgbt_ws (no tartandriver_ws needed).
# Builds nothing here; assumes `colcon build --symlink-install --packages-up-to ros_torch_converter`
# was run in tartan_rgbt_ws. Mounts the workspace at the container path the configs' calib_file
# expects (/home/tartandriver/tartan_rgbt_ws) and mounts /media for bags + output.
#
# Usage:
#   run_extract_docker.sh <config.yaml> <bag_src_dir> <dst_dir>
# Example (10 Hz):
#   run_extract_docker.sh \
#     /home/tartandriver/tartan_rgbt_ws/src/extract/ros_torch_converter/config/kitti_config/tartan_rgbt.yaml \
#     /media/share/.../bags/day1/<seq> /media/share/.../extracted/day1/<seq>
set -euo pipefail

HOST_WS="/home/parv/phd/tartan_rgbt_ws"
CTR_WS="/home/tartandriver/tartan_rgbt_ws"          # must match calib_file path in the configs
IMAGE="tartandriver/main:latest"
CONFIG="${1:?config.yaml (container path under $CTR_WS/...) required}"
SRC="${2:?bag src dir required}"
DST="${3:?dst dir required}"
U=$(id -u); G=$(id -g)

docker run --rm --gpus all --user "$U:$G" -e HOME=/home/parv \
  -v "$HOST_WS:$CTR_WS" -v /media:/media -v /home:/home \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "$IMAGE" bash -lc "
    source /opt/ros/humble/setup.bash
    source $CTR_WS/install/local_setup.bash
    export PYTHONPATH=$CTR_WS/src/extract/ros_torch_converter:\$PYTHONPATH
    cd $CTR_WS/src/extract/ros_torch_converter
    python3 scripts/ros2bag_2_kitti.py --config '$CONFIG' --src_dir '$SRC' --dst_dir '$DST' --force --no_plot
  "
