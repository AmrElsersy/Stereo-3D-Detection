import argparse
import os
import numpy as np
import scipy.misc as ssc
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images')
    
    parser.add_argument('--disparity_dir', type=str,
                        default='./input')
    parser.add_argument('--save_dir', type=str,
                        default='./results')
    
    args = parser.parse_args()
    
    assert os.path.isdir(args.disparity_dir)
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'npy']
    disps = sorted(disps)

    for fn in disps:
        disp_map = np.load(args.disparity_dir + '/' + fn)
        disp_map = (disp_map*256).astype(np.uint16)/256.
        cv2.imwrite(args.save_dir + '/' + fn[:-4] + '.png', disp_map)

