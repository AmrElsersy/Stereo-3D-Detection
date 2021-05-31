import argparse
import os, sys
import skimage
import imageio
import numpy as np
import scipy.misc as ssc
import cv2
from PIL import Image
import torch
src_dir = os.path.dirname(os.path.realpath(__file__))

while not src_dir.endswith("Stereo-3D-Detection"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Models.AnyNet.preprocessing import kitti_util

def generate_dispariy_from_velo(pc_velo, height, width, calib):
    pts_2d = torch.tensor(calib.project_velo_to_image(pc_velo))
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = torch.zeros((height, width)) - 1
    imgfov_pts_2d = torch.round(imgfov_pts_2d).long()
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[(imgfov_pts_2d[i, 1]).long(), (imgfov_pts_2d[i, 0]).long()] = depth
    baseline = 0.54
    disp_map = (calib.f_u * baseline) / depth_map
    return disp_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--datapath', type=str, default='../data/kitti/training/')
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    assert os.path.isdir(args.datapath)
    lidar_dir = args.datapath + '/velodyne/'
    calib_dir = args.datapath + '/calib/'
    image_dir = args.datapath + '/image_2/'
    disparity_dir = args.datapath + "/generated_disp" + '/disp_occ_0/'
    disparity_npy_dir = args.datapath + "/generated_disp" + '/disp_occ_0_npy/'


    assert os.path.isdir(lidar_dir)
    assert os.path.isdir(calib_dir)
    assert os.path.isdir(image_dir)

    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)
    
    if not os.path.isdir(disparity_npy_dir):
        os.makedirs(disparity_npy_dir)

    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
    lidar_files = sorted(lidar_files)

    for i, fn in enumerate(lidar_files):
        if (not args.limit == -1) and (args.limit == i):
            break
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 3))[:, :3]
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = imageio.imread(image_file)
        height, width = image.shape[:2]
        lidar = torch.tensor(lidar)
        disp_map = generate_dispariy_from_velo(lidar, height, width, calib)
        disp_map = disp_map.numpy()
        np.save(disparity_npy_dir + predix, disp_map)
        
        disp_map = (disp_map*256).astype(np.uint16)
        saved = Image.fromarray(disp_map)
        saved.save(disparity_dir + '/' + predix + '.png')
        print('Finish Disparity {}'.format(predix))
