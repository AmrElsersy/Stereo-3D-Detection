import argparse
import os, time,sys
import PIL 
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

src_dir = os.path.dirname(os.path.realpath(__file__))

while not src_dir.endswith("Stereo-3D-Detection"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Models.AnyNet.preprocessing.kitti_util import Calibration 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def project_disp_to_points(calib, disp, max_high):
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
    parser.add_argument('--datapath', type=str, default='data/kitti/training')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')
    parser.add_argument('--limit', type=int, default=-1)

    args = parser.parse_args()

    disparity_dir = os.path.join(src_dir, args.datapath, "disp_occ_0")
    calib_dir = os.path.join(src_dir, args.datapath, "calib")
    save_dir = os.path.join(src_dir, args.datapath, "generated_lidar", "velodyne")

    assert os.path.isdir(disparity_dir)
    assert os.path.isdir(calib_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    disps = [x for x in os.listdir(disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)

    for i, fn in enumerate(disps):
        if (not args.limit == -1) and (args.limit == i):
            break
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        calib = Calibration(calib_file)
        if fn[-3:] == 'png':
            img = Image.open(disparity_dir + '/' + fn)
            disp_map = transforms.ToTensor()(img).squeeze()
            print(disp_map.shape)
        elif fn[-3:] == 'npy':
            disp_map = torch.tensor(np.load(disparity_dir + '/' + fn))
        else:
            assert False
        if not args.is_depth:
            disp_map = disp_map.float()
            lidar = project_disp_to_points(calib, disp_map, args.max_high)
        else:
            disp_map = disp_map.float()
            lidar = project_depth_to_points(calib, disp_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{}.bin'.format(save_dir, predix))
        print('Generated Lidar {}'.format(predix))
