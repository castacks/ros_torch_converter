import os
import tqdm
import shutil
import argparse

from tartandriver_utils.os_utils import kitti_n_frames

from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.image import ImageTorch, FeatureImageTorch
from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch

def setup_dst_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for subfile in os.listdir(src_dir):
        src_path = os.path.join(src_dir, subfile)
        dst_path = os.path.join(dst_dir, subfile)
        if os.path.isdir(src_path):
            os.makedirs(dst_path, exist_ok=True)
        else:
            shutil.copy(src_path, dst_path)

def copy_modality_to_h5(src_dir, dst_dir, modality, dtype):
    src_dir = os.path.join(src_dir, modality)
    dst_dir = os.path.join(dst_dir, modality)

    N = kitti_n_frames(src_dir)

    ## used to generate subset
    # for i, ii in tqdm.tqdm(enumerate(range(500, 600))):
    #     vgt = dtype.from_kitti(src_dir, ii)
    #     vgt.to_kitti(dst_dir, i)
    #     vgt2 = dtype.from_kitti(dst_dir, i)

    #     assert vgt == vgt2

    for i in tqdm.tqdm(range(N)):
        vgt = dtype.from_kitti(src_dir, i)
        vgt.to_kitti(dst_dir, i)
        vgt2 = dtype.from_kitti(dst_dir, i)

        assert vgt == vgt2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='the orig run dir')
    parser.add_argument('--dst_dir', type=str, required=True, help='the place to save outputs')
    args = parser.parse_args()

    setup_dst_dir(args.src_dir, args.dst_dir)

    modalities = [x for x in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, x))]

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

    for i, (modality, dtype) in enumerate(modalities.items()):
        print(f'cp modality {modality} {i+1}/{len(modalities)}')
        copy_modality_to_h5(args.src_dir, args.dst_dir, modality, dtype)