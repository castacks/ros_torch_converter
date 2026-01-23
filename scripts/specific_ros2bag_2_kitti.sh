declare -a dirs=(
# "20251202/teleop/00_warehouse_to_warehousefig8"
# "20251202/teleop/01_fig8_snow_skydio"
"20251202/teleop/02_fig8_explore"
# "20251202/teleop/03_to_horseshoe"
# "20251202/teleop/04_down_to_turnpike"
# "20251202/teleop/05_fence_to_warehouse"
)


# src_dir_path=${TARTANDRIVER_ROSBAG_DIR}/
src_dir_path="/home/tartandriver/rosbags/"
dst_dir_path=${TARTANDRIVER_DATA_DIR}/dynamics/yamaha_kitti/

# Check if all source paths exist
missing_paths=()
echo "Checking source paths..."

for i in "${dirs[@]}"
do
    full_path="${src_dir_path}${i}"
    if [ ! -d "$full_path" ]; then
        missing_paths+=("$full_path")
        echo "MISSING: $full_path"
    else
        echo "FOUND:   $full_path"
    fi
done

# Display results
echo ""
if [ ${#missing_paths[@]} -eq 0 ]; then
    echo "All ${#dirs[@]} source paths found!"
else
    echo "${#missing_paths[@]} out of ${#dirs[@]} paths are missing:"
    for path in "${missing_paths[@]}"; do
        echo "   - $path"
    done
fi

# Ask user to continue
# echo ""
# read -p "Do you want to continue with the processing? (y/n): " -n 1 -r
# echo ""

# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Exiting..."
#     exit 1
# fi

echo "Starting processing..."
echo ""

# Original processing loop
for i in "${dirs[@]}"
do
    # tmp_kitti_path=/local/kitti/${i}
    # tmp_bag_path=/local/bag/${i}
    src_path=${src_dir_path}${i}
    dst_path=${dst_dir_path}${i}
    echo "Processing: ${i}"
    echo ${src_path}
    echo ${dst_path}
    # echo ${tmp_bag_path}
    # echo ${tmp_kitti_path}

    # echo "Copying data to tmp: ${tmp_bag_path}..."
    # mkdir -p ${tmp_bag_path}
    # time cp -r ${src_path}/* ${tmp_bag_path}/

    echo "Executing ros2bag_2_kitti.py..."
    time python3 ros2bag_2_kitti.py --config ../config/kitti_config/tartandrive.yaml \
        --src_dir "${src_path}" \
        --dst_dir "${dst_path}" \
        --force

    # echo "Copying data from tmp to dst: ${dst_path}..."
    # mkdir -p ${dst_path}
    # time cp -r ${tmp_kitti_path}/* ${dst_path}/
done

echo "Processing complete!"
