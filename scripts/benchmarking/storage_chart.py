import os
import time
import argparse

import matplotlib.pyplot as plt

from tartandriver_utils.os_utils import kitti_n_frames

from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.image import ImageTorch, FeatureImageTorch
from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch

#https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
def dir_size(fp):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(fp):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / (1000**3) #bytes->GB

def load_speed(fp, modality, dtype):
    base_dir = os.path.join(fp, modality)
    N = min(100, kitti_n_frames(base_dir))
    t1 = time.time()
    for i in range(N):
        dpt = dtype.from_kitti(base_dir, i)
    t2 = time.time()
    return (t2 - t1) / N

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='path to the run dir to profile')
    parser.add_argument('--title', type=str, required=False)
    args = parser.parse_args()

    subdirs = os.listdir(args.run_dir)
    
    ##debug
    modalities = {
        'voxel_map': VoxelGridTorch,
        'voxel_map_inpaint': VoxelGridTorch,
        'coord_voxel_map': VoxelGridTorch,
        'bev_map_reduce': BEVGridTorch,
        'bev_map_inpaint_reduce': BEVGridTorch,
        'image': ImageTorch,
        'feature_image': FeatureImageTorch,
        'radio3_siglip2_feature_image_full': FeatureImageTorch
    }

    modalities = {k:v for k,v in modalities.items() if k in subdirs}

    sizes = {k:dir_size(os.path.join(args.run_dir, k)) for k in modalities.keys()}
    speeds = {k:load_speed(args.run_dir, k, v) for k,v in modalities.items()}

    ## lol thanks chatgpt
    fig, axs = plt.subplots(1, 2, figsize=(32, 12))

    if args.title is not None:
        fig.suptitle(args.title)

    total_storage = sum(sizes.values())
    dpt_storage = total_storage / kitti_n_frames(args.run_dir)
    
    axs[0].set_title(f"{args.run_dir} Storage (total: {total_storage:.2f}GB, {1000*dpt_storage:.0f}MB/dpt)")
    axs[0].pie([x + 1e-8 for x in sizes.values()], labels=[f"{k} ({v:.2f}GB)" for k,v in sizes.items()], wedgeprops=dict(width=0.4))
    
    axs[1].set_title(f"{args.run_dir} Speed (total: {sum(speeds.values()):.2f}s)")
    axs[1].pie([x + 1e-8 for x in speeds.values()], labels=[f"{k} ({v:.2f}s)" for k,v in speeds.items()], wedgeprops=dict(width=0.4))
    
    plt.show()