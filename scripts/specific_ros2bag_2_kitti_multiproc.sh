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
dst_dir_path=${TARTANDRIVER_DATA_DIR}/dynamics/yamaha_kitti/testspeed

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
echo "src: ${src_path}"
echo "dst: ${dst_path}"

# Ask user to continue
echo ""
read -p "Do you want to continue with the processing? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 1
fi

echo "Starting processing..."
echo ""

# Original processing loop
for i in "${dirs[@]}"
do
    # Uncomment for airlab storage proc speeds
    # src_path=${src_dir_path}${i}
    # dst_path=${dst_dir_path}${i}
    # echo "Processing: ${i}"

    # echo "Executing ros2bag_2_kitti.py..."
    # time python3 ros2bag_2_kitti_multiproc.py --config ../config/kitti_config/tartandrive.yaml \
    #     --src_dir "${src_path}" \
    #     --dst_dir "${dst_path}" \
    #     --force \
    #     --num_workers 12 \
    #     --color

    # Uncomment for local copy speeds
    tmp_kitti_path=${TARTANDRIVER_SCRATCH_DIR}/dataset/${i}
    tmp_bag_path=${TARTANDRIVER_ROSBAG_DIR}/${i}

    echo "Copying data to local: ${tmp_bag_path}..."
    mkdir -p ${tmp_bag_path}
    time rsync -avP ${src_path}/* ${tmp_bag_path}

    echo "Executing ros2bag_2_kitti.py..."
    time python3 ros2bag_2_kitti_multiproc.py --config ../config/kitti_config/tartandrive.yaml \
        --src_dir "${src_path}" \
        --dst_dir "${dst_path}" \
        --force \
        --num_workers 12 \
        --color
    
    echo "Copying data from tmp to dst: ${dst_path}..."
    mkdir -p ${dst_path}
    time rsync -avP ${tmp_kitti_path}/* ${dst_path}/

    echo "Clean up clean up everybody everywhere"
    time rm -r ${tmp_bag_path} ${tmp_kitti_path}
done

echo "Processing complete!"
