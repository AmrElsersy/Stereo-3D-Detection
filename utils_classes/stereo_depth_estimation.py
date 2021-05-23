import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import time
import numpy as np
from PIL import Image

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

        left_img = left_img.crop((w - 1200, h - 352, w, h))
        right_img = right_img.crop((w - 1200, h - 352, w, h))

        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)

        left_img = left_img.clone().detach().reshape(1, *left_img.size())
        right_img = right_img.clone().detach().reshape(1, *right_img.size())
        return left_img, right_img

    def predict(self, imgL, imgR, calib_path):

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
        
        return psuedo_pointcloud

    def stereo_to_disparity(self, imgL, imgR):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        with torch.no_grad():
            outputs = self.model(imgL, imgR)
            disp_map = torch.squeeze(outputs[-1], 1)
            return disp_map
    
    def disparity_to_BEV(self, disp_map, calib_path):
        disp_map = disp_map.float()
        disp_map = disp_map[0]
        if not calib_path == self.calib_path:
            self.calib = Calibration(calib_path)
        # Disparity to point cloud convertor
        lidar = self.gen_lidar(disp_map)
        # Sparsify point cloud convertor
        sparse_points = self.gen_sparse_points(lidar)

        filtered = get_filtered_lidar(sparse_points, cnf.boundary)

        bev = makeBEVMap(filtered, cnf.boundary)
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

    
