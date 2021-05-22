import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import time
import numpy as np
from Models.AnyNet.preprocessing.generate_lidar import project_disp_to_points, Calibration
from Models.AnyNet.preprocessing.kitti_sparsify import pto_ang_map
from Models.AnyNet.models.anynet import AnyNet
from visualization.KittiUtils import *
from utils_classes.stereo_preprocessing import StereoPreprocessing

# ========================= Stereo =========================
class Stereo_Depth_Estimation:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model()
        self.preprocesiing = StereoPreprocessing()

    def load_model(self):
        model = AnyNet(self.args)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(self.args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    def predict(self, imgL, imgR, calib_path, return_disparity=False):

        start = time.time()
        imgL, imgR = self.preprocesiing.preprocess(imgL, imgR)
        end = time.time()
        print(f"Time for pre-processing: {1000 * (end - start)} ms")

        disparity, start = self.stereo_to_disparity(imgL, imgR)
        psuedo_pointcloud = self.disparity_to_pointcloud(disparity, calib_path)
        end = time.time()
        print(f"Time for post processing: {1000 * (end - start)} ms")

        if return_disparity:
            return disparity, psuedo_pointcloud

        return psuedo_pointcloud

    def stereo_to_disparity(self, imgL, imgR):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        with torch.no_grad():
            start = time.time()
            outputs = self.model(imgL, imgR)
            end = time.time()
            print(f"Time for stereo: {1000 * (end - start)} ms")
            start = time.time()
            output3 = torch.squeeze(outputs[-1], 1)
            image = output3.cpu()
            img_cpu = np.asarray(image)
            disp_map = img_cpu[0, :, :]
            disp_map = (disp_map).astype(np.float32)
            return disp_map, start
    
    def disparity_to_pointcloud(self, disp_map, calib_path):
        calib = Calibration(calib_path)
        # Disparity to point cloud convertor
        lidar = self.gen_lidar(disp_map, calib)
        # Sparsify point cloud convertor
        sparse_points = self.gen_sparse_points(lidar)
        return sparse_points

    def gen_lidar(self, disp_map, calib, max_high=1):
        lidar = project_disp_to_points(calib, disp_map, max_high)
        lidar = lidar.astype(np.float32)
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
        sparse_points = sparse_points.astype(np.float32)
        return sparse_points
