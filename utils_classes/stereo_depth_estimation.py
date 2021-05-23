import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from Models.AnyNet.preprocessing.generate_lidar import project_disp_to_points, Calibration
from Models.AnyNet.preprocessing.kitti_sparsify import pto_ang_map
from Models.AnyNet.dataloader import preprocess
from Models.AnyNet.models.anynet import AnyNet

import Models.SFA.config.kitti_config as cnf
from Models.SFA.data_process.kitti_data_utils import get_filtered_lidar
from Models.SFA.data_process.kitti_bev_utils import makeBEVMap

from visualization.KittiUtils import *

def default_loader(path):
    return Image.open(path).convert('RGB')

# ========================= Stereo =========================
class Stereo_Depth_Estimation:
    def __init__(self, cfgs, loader=default_loader):
        self.cfgs = cfgs
        self.model = self.load_model()
        self.loader = loader
        self.calib_path = None
        self.calib = None

    def load_model(self):
        model = AnyNet(self.cfgs)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(self.cfgs.pretrained_anynet)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model
    
    def preprocess(self, left_img, right_img):
        # left_img = left_img.convert('RGB')
        # right_img = right_img.convert('RGB')
        left_img = Image.fromarray(np.uint8(left_img)).convert('RGB')
        right_img = Image.fromarray(np.uint8(right_img)).convert('RGB')
        w, h = left_img.size

        left_img =  left_img.crop((w - 1200, h - 352, w, h))
        right_img = right_img.crop((w - 1200, h - 352, w, h))

        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img).cuda()
        right_img = processed(right_img).cuda()

        left_img = left_img.reshape(1, *left_img.size())
        right_img = right_img.reshape(1, *right_img.size())
        return left_img, right_img

    def predict(self, imgL, imgR, calib_path, printer=False):
        if printer:
            start = time.time()
            imgL, imgR = self.preprocess(imgL, imgR)
            end = time.time()
            print(f"Time for pre-processing: {1000 * (end - start)} ms")

            start = time.time()
            disparity = self.stereo_to_disparity(imgL, imgR)
            end = time.time()
            print(f"Time for stereo: {1000 * (end - start)} ms")

            start = time.time()
            psuedo_pointcloud = self.disparity_to_BEV(disparity, calib_path)
            end = time.time()
            print(f"Time for post processing: {1000 * (end - start)} ms")
        else:
            imgL, imgR = self.preprocess(imgL, imgR)
            disparity = self.stereo_to_disparity(imgL, imgR)
            psuedo_pointcloud = self.disparity_to_BEV(disparity, calib_path)

        return psuedo_pointcloud

    def stereo_to_disparity(self, imgL, imgR):
        imgL = imgL.float()
        imgR = imgR.float()
        with torch.no_grad():
            outputs = self.model(imgL, imgR)
            disp_map = torch.squeeze(outputs[-1], 1)[0].float()
            return disp_map
    
    def disparity_to_BEV(self, disp_map, calib_path):
        if not calib_path == self.calib_path:
            self.calib = Calibration(calib_path)
        # Disparity to point cloud convertor
        lidar = self.gen_lidar(disp_map)
        # Sparsify point cloud convertor
        sparse_points = self.gen_sparse_points(lidar)

        filtered = get_filtered_lidar(sparse_points, cnf.boundary)

        bev = self.makeBEVMap(filtered, cnf.boundary)
        bev = torch.from_numpy(bev)
        bev = torch.unsqueeze(bev, 0)
        bev = bev.to(self.cfgs.device, non_blocking=True).float()
        return bev

    def gen_lidar(self, disp_map, max_high=1):
        lidar = project_disp_to_points(self.calib, disp_map, max_high)
        return lidar

    def gen_sparse_points(self, pc_velo, H=64, W=512, D=700, slice=1):
        valid_inds =    (pc_velo[:, 0] < 120)    & \
                        (pc_velo[:, 0] >= 0)     & \
                        (pc_velo[:, 1] < 50)     & \
                        (pc_velo[:, 1] >= -50)   & \
                        (pc_velo[:, 2] < 1.5)    & \
                        (pc_velo[:, 2] >= -2.5)
        pc_velo = pc_velo[valid_inds]
        sparse_points = pto_ang_map(pc_velo, H=H, W=W, slice=slice)
        return sparse_points

    def makeBEVMap(PointCloud, boundary):
        Height = cnf.BEV_HEIGHT + 1
        Width = cnf.BEV_WIDTH + 1

        # Discretize Feature Map
        PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

        # sort-3times
        sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
        PointCloud = PointCloud[sorted_indices]
        _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        PointCloud_top = PointCloud[unique_indices]

        # Height Map, Intensity Map & Density Map
        heightMap = np.zeros((Height, Width))
        intensityMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
        heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

        normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
        intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = 1
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
        RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
        RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

        return RGB_Map

    
