import argparse
import os
import skimage
import skimage.io
import numpy as np
import scipy.misc as ssc
import cv2
from PIL import Image
import kitti_util

def generate_dispariy_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    baseline = 0.54
    disp_map = (calib.f_u * baseline) / depth_map
    return disp_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/velodyne/'
    calib_dir = args.data_path + '/calib/'
    image_dir = args.data_path + '/image_2/'
    disparity_dir = args.data_path + '/disp_occ_0/'
    disparity_npy_dir = args.data_path + '/disp_occ_0_npy/'


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
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3]
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = ssc.imread(image_file)
        height, width = image.shape[:2]
        disp_map = generate_dispariy_from_velo(lidar, height, width, calib)
        np.save(disparity_npy_dir + '/' + predix, disp_map)

        # disp_map = np.clip(disp_map, 0, 2**16)
        # disp_map = (disp_map - np.min(disp_map)) / (np.max(disp_map) - np.min(disp_map))
        # disp_map = (disp_map*256).astype(np.uint8)
        # saved = Image.fromarray(disp_map)
        # saved.save(disparity_dir + '/' + predix + '.png')
        print('Finish Disparity {}'.format(predix))
