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
from visualization.BEVutils import *

def default_loader(path):
    return Image.open(path).convert('RGB')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        lidar = lidar.to(device)

        # Sparsify point cloud convertor
        sparse_points = self.gen_sparse_points(lidar)
        print(sparse_points.device)

        filtered = self.get_filtered_lidar(sparse_points, cnf.boundary) 

        # our implementation (from )
        print(filtered.device)
        bev = self.makeBEVMap(filtered)
        print(bev.device)

        # SFA implementation
        # bev = self.makeBEVMap2(filtered) 

        # numpy implementation ( from 58 to 66 ms)
        # bev = makeBEVMap(filtered, cnf.boundary)
        # bev = torch.from_numpy(bev)

        # visualize
        # import cv2
        # b = bev.permute((1,2,0)).cpu().numpy()
        # cv2.imshow('bev', b)
        # cv2.waitKey(0)

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


    def makeBEVMap(self, pointcloud):
        torch.cuda.empty_cache()
        pointcloud = pointcloud.to(device)

        # sort by z ... to get the maximum z when using unique 
        # (as unique function gets the first unique elemnt so we attatch it with max value)
        z_indices = torch.argsort(pointcloud[:,2])
        pointcloud = pointcloud[z_indices]

        MAP_HEIGHT = cnf.BEV_HEIGHT + 1
        MAP_WIDTH  = cnf.BEV_WIDTH  + 1

        height_map    = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32, device=device) # max z
        intensity_map = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32, device=device) # intensity (contains reflectivity or 1 if not supported)
        density_map   = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32, device=device) # density of the mapped 3D points to a the pixel

        x_bev = torch.tensor( pointcloud[:, 0] * descretization_x, dtype=torch.long).to(device)
        y_bev = torch.tensor((cnf.BEV_WIDTH/2) + pointcloud[:, 1] * descretization_y, dtype=torch.long).to(device)
        z_bev = pointcloud[:, 2] # float32, cuda
                    
        xy_bev = torch.stack((x_bev, y_bev), dim=1)

        # counts.shape  (n_unique_elements,) .. counts is count of repeate times of each unique element (needed for density)
        xy_bev_unique, inverse_indices, counts = torch.unique(xy_bev, return_counts=True, return_inverse=True, dim=0)

        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
        inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
        indices = inverse_indices.new_empty(xy_bev_unique.size(0)).scatter_(0, inverse_indices, perm)

        # 1 or reflectivity if supported
        intensity_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = 1

        # points are sorted by z, so unique indices (first found indices) is the max z
        height_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = z_bev[indices] / max_height

        density_map[xy_bev_unique[:,0], xy_bev_unique[:,1]] = torch.minimum(
            torch.ones_like(counts), 
            torch.log(counts.float() + 1)/ np.log(64)
            )

        RGB_Map = torch.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH), dtype=torch.float32, device=device)
        RGB_Map[2, :, :] = density_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = height_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
        RGB_Map[0, :, :] = intensity_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

        return RGB_Map
        

    # THIS IS NOTTTT WORKING YET
    def makeBEVMap2(self, pointcloud):
        torch.cuda.empty_cache()
        pointcloud = pointcloud.to(device)

        Height = cnf.BEV_HEIGHT + 1
        Width = cnf.BEV_WIDTH + 1

        # Discretize Feature Map        
        pointcloud[:, 0] = torch.tensor(torch.floor(pointcloud[:, 0] / cnf.DISCRETIZATION))
        pointcloud[:, 1] = torch.tensor(torch.floor(pointcloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

        z_indices = torch.argsort(pointcloud[:,2])
        pointcloud = pointcloud[z_indices]

        xy_bev_unique, inverse_indices, counts = torch.unique(pointcloud[:, 0:2], return_counts=True, return_inverse=True, dim=0)

        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
        inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
        indices = inverse_indices.new_empty(xy_bev_unique.size(0)).scatter_(0, inverse_indices, perm)

        pointcloud_top = pointcloud[indices]
        pointcloud_top = pointcloud_top.long()

        # Height Map, Intensity Map & Density Map
        heightMap    = torch.zeros((Height, Width), dtype=torch.float32, device=device)
        intensityMap = torch.zeros((Height, Width), dtype=torch.float32, device=device)
        densityMap   = torch.zeros((Height, Width), dtype=torch.float32, device=device)

        # some important problem is image coordinate is (y,x), not (x,y)
        heightMap[pointcloud_top[:, 0], pointcloud_top[:, 1]] = pointcloud_top[:, 2] / max_height

        intensityMap[pointcloud_top[:, 0], pointcloud_top[:, 1]] = 1
        densityMap[pointcloud_top[:, 0], pointcloud_top[:, 1]] = torch.minimum(
            torch.ones_like(counts), 
            torch.log(counts.float() + 1)/ np.log(64)
            )

        RGB_Map = torch.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH), dtype=torch.float32, device=device)
        RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
        RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

        return RGB_Map

    
    def get_filtered_lidar(self, lidar, boundary):
        minX = boundary['minX']
        maxX = boundary['maxX']
        minY = boundary['minY']
        maxY = boundary['maxY']
        minZ = boundary['minZ']
        maxZ = boundary['maxZ']

        # Remove the point out of range x,y,z
        mask = torch.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                        (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                        (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
        lidar = lidar[mask]
        lidar[:, 2] = lidar[:, 2] - minZ

        return lidar
