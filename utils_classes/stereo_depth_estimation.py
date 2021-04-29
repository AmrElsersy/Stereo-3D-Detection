import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import time
import numpy as np
from AnyNet.preprocessing.generate_lidar import project_disp_to_points, Calibration
from AnyNet.preprocessing.kitti_sparsify import pto_ang_map
from AnyNet.models.anynet import AnyNet
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

        imgL, imgR = self.preprocesiing.preprocess(imgL, imgR)

        disparity = self.stereo_to_disparity(imgL, imgR)
        psuedo_pointcloud = self.disparity_to_pointcloud(disparity, calib_path)

        if return_disparity:
            return disparity[-1] , psuedo_pointcloud

        return psuedo_pointcloud

    def stereo_to_disparity(self, imgL, imgR):
        self.model.eval()

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()

        with torch.no_grad():
            startTime = time.time()
            outputs = self.model(imgL, imgR)
            return outputs
    
    def disparity_to_pointcloud(self, disparity, calib_path):
        # get the last disparity (best accuracy)
        output = disparity[-1]
        output = torch.squeeze(output ,1)

        img_cpu = np.asarray(output.cpu())
        disp_map = img_cpu[0, :, :]
        calib = Calibration(calib_path)
        lidar = gen_lidar(disp_map, calib)
        sparse_points = gen_sparse_points(lidar)
        return sparse_points

def gen_lidar(disp_map, calib, max_high=1):
    disp_map = (disp_map*255).astype(np.float32)/255.
    lidar = project_disp_to_points(calib, disp_map, max_high)
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)
    return lidar

def gen_sparse_points(lidar, H=64, W=512, D=700, slice=1):
    pc_velo = lidar.reshape((-1, 4))
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