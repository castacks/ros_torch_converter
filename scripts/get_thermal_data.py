'''
For any bag before 3/15/2025 without /thermal_left/image_processed topics,
run this post-processing script to generate processed 8-bit thermal from 16-bit thermal images.
'''

import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm
import shutil
import yaml

def read_kalibr_stereo(config):
    '''Read a typical kalibr stereo calibration from config'''
    calib_data = config['calib']
    
    calib_dict = {}
    fx, fy, cx, cy = calib_data['cam0']['intrinsics']
    K_left = np.array([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]])
    distort_left = np.array(calib_data['cam0']['distortion_coeffs'])
    calib_dict['K_left'] = K_left
    calib_dict['distort_left'] = distort_left

    fx, fy, cx, cy = calib_data['cam1']['intrinsics']
    K_right = np.array([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]])
    distort_right = np.array(calib_data['cam1']['distortion_coeffs'])
    calib_dict['K_right'] = K_right
    calib_dict['distort_right'] = distort_right
    
    calib_dict['T_right2left'] = np.array(calib_data['cam1']['T_cn_cnm1']).reshape(4,4)

    return calib_dict

def rectify_image(image, intrinsics, distortion):
    rectified_image = cv2.undistort(image, intrinsics, distortion)
    return rectified_image

def stereo_rectify(left, right, calib_dict) -> np.ndarray:
    """
    Stereo rectify the left and right images

    Returns:
        left_rect: Rectified left image
        right_rect: Rectified right image
        K_left: New Left camera intrinsic matrix
        K_right: New Right camera intrinsic matrix
    """
    width, height = left.shape[1], left.shape[0]
    T_left2right = np.linalg.inv(calib_dict['T_right2left'])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(calib_dict['K_left'], calib_dict['distort_left'], 
                        calib_dict['K_right'], calib_dict['distort_right'], (width, height),
                        T_left2right[:3, :3], T_left2right[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    
    undistort_map_left = cv2.initUndistortRectifyMap(calib_dict['K_left'], calib_dict['distort_left'], R1, P1, (width, height), cv2.CV_32FC1)
    undistort_map_right = cv2.initUndistortRectifyMap(calib_dict['K_right'], calib_dict['distort_right'], R2, P2, (width, height), cv2.CV_32FC1)
    
    left_rect = cv2.remap(left, undistort_map_left[0], undistort_map_left[1], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, undistort_map_right[0], undistort_map_right[1], cv2.INTER_LINEAR)
    
    K_left = P1[:3, :3]
    K_right = P2[:3, :3]
    return left_rect, right_rect, K_left, K_right

def enhance_image(image):
    # Expects 8-bit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe.apply(image)
    bilateral = cv2.bilateralFilter(clahe, 5, 20, 15)
    return bilateral

def process_image(image_in, type):
    # converts 16bit to 8bit
    if type == "minmax":
        image_out = (image_in - np.min(image_in)) / (np.max(image_in) - np.min(image_in)) * 255
    elif "firestereo" in type:
        if type=="firestereo" and np.max(image_in) < 35000:
            image_out = (image_in - np.min(image_in)) / (np.max(image_in) - np.min(image_in)) * 255
        else: #firestereo_v2
            im_srt = np.sort(image_in.reshape(-1))
            upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
            lower_bound = im_srt[round(len(im_srt) * 0.01)]

            img = image_in
            img[img < lower_bound] = lower_bound
            img[img > upper_bound] = upper_bound
            image_out = ((img - lower_bound) / (upper_bound - lower_bound)) * 255.0
            image_out = image_out.astype(np.uint8)
    else:
        image_out = image_in / 255

    return image_out.astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='/path/to/dataset')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config, 'r'))
    
    calib_dict = read_kalibr_stereo(config)
    
    left_imgs = sorted(glob.glob(os.path.join(args.dataset, config['left_dir'], '*.png')))
    right_imgs = sorted(glob.glob(os.path.join(args.dataset, config['right_dir'], '*.png')))
    
    print(f"Found {len(left_imgs)} left images and {len(right_imgs)} right images")
    assert len(left_imgs) == len(right_imgs), "Left and right image directories must have the same number of images"

    left_out_dir = os.path.join(args.dataset, f"{config['left_dir']}_{config['output_suffix']}")
    right_out_dir = os.path.join(args.dataset, f"{config['right_dir']}_{config['output_suffix']}")
    os.makedirs(left_out_dir, exist_ok=True)
    os.makedirs(right_out_dir, exist_ok=True)
    
    left_timestamps = os.path.join(args.dataset, config['left_dir'], 'timestamps.txt')
    right_timestamps = os.path.join(args.dataset, config['right_dir'], 'timestamps.txt')
    shutil.copy(left_timestamps, os.path.join(left_out_dir, 'timestamps.txt'))
    shutil.copy(right_timestamps, os.path.join(right_out_dir, 'timestamps.txt'))

    for left_img_path, right_img_path in tqdm(zip(left_imgs, right_imgs), total=len(left_imgs)):
        left_filename = os.path.basename(left_img_path)
        right_filename = os.path.basename(right_img_path)
        
        assert left_filename == right_filename, f"Image names don't match: {left_filename} vs {right_filename}"
        
        left_img = cv2.imread(left_img_path, cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_UNCHANGED)
        
        if left_img is None or right_img is None:
            print(f"Warning: Could not load images {left_filename}")
            continue
        
        if config['rectify']:
            left_img, right_img, K_left, K_right = stereo_rectify(left_img, right_img, calib_dict)
        
        if config['process'] is not None:
            left_img = process_image(left_img, config['process'])
            right_img = process_image(right_img, config['process'])
        
        if config['enhance']:
            left_img = enhance_image(left_img)
            right_img = enhance_image(right_img)
        
        left_out_path = os.path.join(left_out_dir, left_filename)
        right_out_path = os.path.join(right_out_dir, right_filename)
        
        cv2.imwrite(left_out_path, left_img)
        cv2.imwrite(right_out_path, right_img)

    print(f'Done processing {len(left_imgs)} stereo thermal image pairs')
    print(f'Output saved to:')
    print(f'  Left: {left_out_dir}')
    print(f'  Right: {right_out_dir}')