'''
For any bag before 3/15/2025 without /thermal_left/image_processed topics,
run this post-processing script to generate processed 8-bit thermal from 16-bit thermal images.
'''

import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import shutil

def rectify_image(image, intrinsics, distortion):
    rectified_image = cv2.undistort(image, intrinsics, distortion)
    return rectified_image

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
    elif type == "firestereo":
        if np.max(image_in) < 35000:
            image_out = (image_in - np.min(image_in)) / (np.max(image_in) - np.min(image_in)) * 255
        else:
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
    parser.add_argument('--img_dir', type=str, required=True, help='/path/to/16bit/thermal/images')
    parser.add_argument('--dst_dir', type=str, default=None, help='/path/to/output/8bit/thermal/images')
    parser.add_argument('--process', type=str, default='firestereo', help='minmax or firestereo')
    parser.add_argument('--rectify', action='store_true', help='rectify image')
    parser.add_argument('--enhance', action='store_true', help='enhance image')
    args = parser.parse_args()
    
    images = [f for f in sorted(os.listdir(args.img_dir)) if f.endswith('.png')]
    if args.dst_dir is None:
        args.dst_dir = f'{args.img_dir}_processed'

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
        print(f'Created folder {args.dst_dir}')
    
    shutil.copy(f'{args.img_dir}/timestamps.txt', f'{args.dst_dir}/timestamps.txt')
    
    intrinsics = np.array([412.42744452, 0.0, 313.38643993,0.0, 412.60673097, 249.37501763, 0.0, 0.0, 1.0]).reshape(3, 3)
    distortion = np.array([-0.367732796157476, 0.12110213142717571, -0.0006255396681117811, 0.00041510869260370575])

    for image in tqdm(images):
        img = cv2.imread(os.path.join(args.img_dir, image), cv2.IMREAD_UNCHANGED)
        img_out = process_image(img, args.process)
        if args.rectify:
            img_out = rectify_image(img_out, intrinsics, distortion)
        if args.enhance:
            img_out = enhance_image(img_out)
        cv2.imwrite(os.path.join(args.dst_dir, image), img_out)

    print(f'Done processing {len(images)} thermal images into {args.dst_dir}')