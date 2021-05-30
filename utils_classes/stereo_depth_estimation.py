from os import error
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from Models.AnyNet.preprocessing.generate_lidar import  Calibration
from Models.AnyNet.models.anynet import AnyNet
import Models.AnyNet.dataloader.preprocess as preprocess


import Models.SFA.config.kitti_config as cnf
from Models.SFA.utils.misc import time_synchronized
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
        if os.path.isfile(self.cfgs.pretrained_anynet):
            model = AnyNet(self.cfgs)
            model = nn.DataParallel(model).cuda()
            checkpoint = torch.load(self.cfgs.pretrained_anynet)
            print("=> loaded anynet pretrained model '{}'".format(self.cfgs.pretrained_anynet))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
        else:
            print("No model at this location '{}'".format(self.cfgs.pretrained_anynet))
            raise error
        return model
    
    def preprocess(self, left_img, right_img):
        normalize = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        left_img = torch.tensor(left_img, dtype=torch.float32, device=device).transpose(0, 1).T
        right_img = torch.tensor(right_img, dtype=torch.float32, device=device).transpose(0, 1).T

        c, h, w = left_img.shape
    
        left_img =  transforms.functional.crop(left_img, h - 352,  w - 1200, h, w)
        right_img =  transforms.functional.crop(right_img, h - 352,  w - 1200, h, w)

        left_img = transforms.Normalize(**normalize)(left_img)
        right_img = transforms.Normalize(**normalize)(right_img)
        # w, h = left_img.size
        # left_img = left_img.crop((w - 1200, h - 352, w, h))
        # right_img = right_img.crop((w - 1200, h - 352, w, h))

        # processed = preprocess.get_transform(augment=False)
        # left_img = processed(left_img)
        # right_img = processed(right_img)
        
        left_img = left_img.unsqueeze(0)
        right_img = right_img.unsqueeze(0)

        return left_img, right_img

    def predict(self, imgL, imgR, calib_path, printer=False):
        if printer:
            start = time_synchronized()
            imgL, imgR = self.preprocess(imgL, imgR)
            end = time_synchronized()
            print(f"Time for pre-processing: {1000 * (end - start)} ms")

            start = time_synchronized()
            disparity = self.stereo_to_disparity(imgL, imgR)
            end = time_synchronized()
            print(f"Time for stereo: {1000 * (end - start)} ms")

            start = time_synchronized()
            psuedo_pointcloud = self.disparity_to_BEV(disparity, calib_path)
            end = time_synchronized()
            print(f"Time for post processing: {1000 * (end - start)} ms")
        else:
            imgL, imgR = self.preprocess(imgL, imgR)
            disparity = self.stereo_to_disparity(imgL, imgR)
            psuedo_pointcloud = self.disparity_to_BEV(disparity, calib_path, printer)

        return psuedo_pointcloud

    def stereo_to_disparity(self, imgL, imgR):
        imgL = imgL.float()
        imgR = imgR.float()
        outputs = self.model(imgL, imgR)
        disp_map = torch.squeeze(outputs[-1], 1)[0].float()
        return disp_map
    
    def disparity_to_BEV(self, disp_map, calib_path, printer=True):
        # start = time_synchronized()
        if not calib_path == self.calib_path:
            self.calib = Calibration(calib_path)
            self.calib_path = calib_path
        # end = time_synchronized()
        # if printer:
        #     print(f"Time for calibrating: {1000 * (end - start)} ms")

        # Disparity to point cloud convertor
        # start = time_synchronized()
        lidar = self.gen_lidar(self.calib, disp_map, self.cfgs.max_high)
        # end = time_synchronized()
        # if printer:
        #     print(f"\nTime for Disparity_LIDAR: {1000 * (end - start)} ms")

        # Sparsify point cloud convertor (Cuda = 2.5 ms instead of 15 ms)
        # start = time_synchronized()
        sparse_points = self.gen_sparse_points(lidar, H=self.cfgs.H, W=self.cfgs.W, slice=self.cfgs.slice)
        # end = time_synchronized()
        # if printer:
        #     print(f"Time for Sparsify: {1000 * (end - start)} ms")

        # filter (no big diffrence between cuda & cpu .. both < 1 ms)
        # start = time_synchronized()
        filtered = self.get_filtered_lidar(sparse_points, cnf.boundary)
        # end = time_synchronized()
        # if printer:
        #     print(f"Time for Filter: {1000 * (end - start)} ms")

        # our implementation (2.5 ms)
        # start = time_synchronized()
        bev = self.makeBEVMap(filtered)
        # end = time_synchronized()
        # if printer:
        #     print(f"Time for BEV: {1000 * (end - start)} ms\n")

        # numpy implementation (10 ms)
        # bev = makeBEVMap(filtered.cpu().numpy(), cnf.boundary)
        # bev = torch.from_numpy(bev).float().cuda()
        # bev = torch.unsqueeze(bev, 0)

        # visualize
        # import cv2
        # b = bev.permute((1,2,0)).cpu().numpy()
        # cv2.imshow('bev', b)
        # cv2.waitKey(0)
        if self.cfgs.with_bev:
            return bev, sparse_points
        
        return bev

    def makeBEVMap(self, pointcloud):
        # sort by z ... to get the maximum z when using unique 
        # (as unique function gets the first unique elemnt so we attatch it with max value)
        z_indices = torch.argsort(pointcloud[:,2])
        pointcloud = pointcloud[z_indices]

        MAP_HEIGHT = cnf.BEV_HEIGHT + 1
        MAP_WIDTH  = cnf.BEV_WIDTH  + 1

        height_map    = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32, device=pointcloud.device)
        intensity_map = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32, device=pointcloud.device)  
        density_map   = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32, device=pointcloud.device)
        x_bev = torch.tensor( pointcloud[:, 0] * descretization_x, dtype=torch.long, device=device)
        y_bev = torch.tensor((cnf.BEV_WIDTH/2) + pointcloud[:, 1] * descretization_y, dtype=torch.long, device=device)
        z_bev = pointcloud[:, 2] # float32, cuda
                    
        xy_bev = torch.stack((x_bev, y_bev), dim=1)

        # counts.shape  (n_unique_elements,) .. counts is count of repeate times of each unique element (needed for density)
        xy_bev_unique, inverse_indices, counts = torch.unique(xy_bev, return_counts=True, return_inverse=True, dim=0)
        

        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
        inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
        indices = inverse_indices.new_empty(xy_bev_unique.size(0), device=device).scatter_(0, inverse_indices, perm)

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

        RGB_Map = torch.unsqueeze(RGB_Map, 0)

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

    def gen_lidar(self, calib, disp, max_high):
        mask = (disp > 0).reshape(-1).long()
        disp = disp.clamp(min=0) + 0.1

        baseline = 0.54
        depth = calib.f_u * baseline / (disp) 
        rows, cols = depth.shape

        c, r = torch.meshgrid(torch.arange(cols, device=device), torch.arange(rows, device=device))
        c = c.T.reshape(-1) * mask
        r = r.T.reshape(-1) * mask
        depth = depth.reshape(-1) * mask
        points = torch.stack([c, r, depth])
        points = points.T

        # (5 - 10 ms)
        cloud = calib.project_image_to_velo(points)

        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        return cloud[valid]

    def gen_sparse_points(self, pc_velo, H=64, W=512, slice=1):
        """
        :param H: the row num of depth map, could be 64(default), 32, 16
        :param W: the col num of depth map
        :param slice: output every slice lines
        """
        valid_inds =    torch.where((pc_velo[:, 0] < 120)    & \
                        (pc_velo[:, 0] >= 0)     & \
                        (pc_velo[:, 1] < 50)     & \
                        (pc_velo[:, 1] >= -50)   & \
                        (pc_velo[:, 2] < 1.5)    & \
                        (pc_velo[:, 2] >= -2.5))
        
        pc_velo = pc_velo[valid_inds]

        def radians(x):
            return x * 0.0174532925

        dtheta = radians(0.4 * 64.0 / H)
        dphi = radians(90.0 / W)

        x, y, z = pc_velo[:, 0], pc_velo[:, 1], pc_velo[:, 2]

        x_y = x**2 + y**2

        d = torch.sqrt(x_y + z ** 2)
        r = torch.sqrt(x_y) 
        d = d.clamp(0.000001)
        r = r.clamp(0.000001)
        phi = radians(45.) - torch.arcsin(y / r)
        phi_ = (phi / dphi).long()
        phi_ = phi_.clamp(min=0, max=W - 1) 

        theta = radians(2.) - torch.arcsin(z / d)
        theta_ = (theta / dtheta).long()
        theta_ = theta_.clamp(min=0, max= H - 1)

        depth_map = - torch.ones((H, W, 3), device= pc_velo.device)
        depth_map[theta_, phi_, 0] = x
        depth_map[theta_, phi_, 1] = y
        depth_map[theta_, phi_, 2] = z
        depth_map = depth_map[0::slice, :, :]
        depth_map = depth_map.reshape((-1, 3))
        depth_map = depth_map[depth_map[:, 0] != -1.0]
        return depth_map