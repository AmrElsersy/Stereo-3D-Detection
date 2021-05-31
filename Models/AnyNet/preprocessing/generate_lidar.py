import argparse
import os, time

import numpy as np
import torch
import scipy.misc as ssc
from .kitti_util import Calibration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def project_disp_to_points(self, calib, disp, max_high):
        mask = (disp > 0).reshape(-1).long()
        disp = disp.clamp(min=0) + 0.1

        baseline = 0.54
        depth = calib.f_u * baseline / (disp) 
        rows, cols = depth.shape

        c, r = torch.meshgrid(torch.arange(cols), torch.arange(rows))
        c = c.T.reshape(-1) * mask
        r = r.T.reshape(-1) * mask
        depth = depth.reshape(-1) * mask
        points = torch.stack([c, r, depth])
        points = points.T

        # (5 - 10 ms)
        cloud = calib.project_image_to_velo(points)

        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        return cloud[valid]

def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = torch.meshgrid(np.arange(cols), np.arange(rows))
    points = torch.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--calib_dir', type=str,
                        default='../path-to-kitti-small/training/calib')
    parser.add_argument('--disparity_dir', type=str,
                        default='../path-to-kitti-small/training/predicted_disparity')
    parser.add_argument('--save_dir', type=str,
                        default='../path-to-kitti-small/training/predicted_velodyne')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')

    args = parser.parse_args()

    assert os.path.isdir(args.disparity_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)

    for fn in disps:
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = Calibration(calib_file)
        # disp_map = ssc.imread(args.disparity_dir + '/' + fn) / 256.
        if fn[-3:] == 'png':
            disp_map = ssc.imread(args.disparity_dir + '/' + fn)
        elif fn[-3:] == 'npy':
            disp_map = torch.tensor(np.load(args.disparity_dir + '/' + fn))
        else:
            assert False
        if not args.is_depth:
            disp_map = disp_map.float()
            # print(np.min(disp_map), np.max(disp_map))
            lidar = project_disp_to_points(calib, disp_map, args.max_high)
        else:
            disp_map = disp_map.float()
            lidar = project_depth_to_points(calib, disp_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar.numpy(), np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))
