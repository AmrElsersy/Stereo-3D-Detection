import argparse
import os
import numpy as np
import scipy.misc as ssc
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images')
    
    parser.add_argument('--main_dir', type=str,
                        default='./input')
    parser.add_argument('--save_dir', type=str,
                        default='./results')
    parser.add_argument('--limit', type=int,
                        default=-1)
    
    args = parser.parse_args()
    
    assert os.path.isdir(args.main_dir)
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    # if not os.path.isdir(args.save_dir + '/clipping_standerlization_255/'):
    #     os.makedirs(args.save_dir + '/clipping_standerlization_255/')
    if not os.path.isdir(args.save_dir + '/standerlization_255/'):
        os.makedirs(args.save_dir + '/standerlization_255/')
    # if not os.path.isdir(args.save_dir + '/standerlization/'):
    #     os.makedirs(args.save_dir + '/standerlization/')

    disps = [x for x in os.listdir(args.main_dir) if x[-3:] == 'npy']
    disps = sorted(disps)

    for i, fn in enumerate(disps):

        if (not args.limit == -1) and (args.limit == i):
            break

        disp = np.load(args.main_dir + '/' + fn)
        disp_map = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
        disp_map = (disp_map*255).astype(np.uint8)
        cv2.imwrite(args.save_dir + '/standerlization_255/' + fn[:-4] + '.png', disp_map)


