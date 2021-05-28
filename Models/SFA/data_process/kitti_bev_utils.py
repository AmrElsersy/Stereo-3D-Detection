"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import math
import os
import sys

import cv2
import numpy as np
import torch

from ..config import kitti_config as cnf
from visualization.BEVutils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def makeBEVMap(pointcloud, boundary):
    # sort by z ... to get the maximum z when using unique 
    # (as unique function gets the first unique elemnt so we attatch it with max value)
    pointcloud = torch.tensor(pointcloud)
    z_indices = torch.argsort(pointcloud[:,2])
    pointcloud = pointcloud[z_indices]

    MAP_HEIGHT = cnf.BEV_HEIGHT + 1
    MAP_WIDTH  = cnf.BEV_WIDTH  + 1

    height_map    = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32)
    intensity_map = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32)  
    density_map   = torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=torch.float32)
    x_bev = torch.tensor( pointcloud[:, 0] * descretization_x, dtype=torch.long)
    y_bev = torch.tensor((cnf.BEV_WIDTH/2) + pointcloud[:, 1] * descretization_y, dtype=torch.long)
    z_bev = pointcloud[:, 2] # float32, cuda
                
    xy_bev = torch.stack((x_bev, y_bev), dim=1)

    # counts.shape  (n_unique_elements,) .. counts is count of repeate times of each unique element (needed for density)
    xy_bev_unique, inverse_indices, counts = torch.unique(xy_bev, return_counts=True, return_inverse=True, dim=0)
    

    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype)
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

    RGB_Map = torch.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH), dtype=torch.float32)
    RGB_Map[2, :, :] = density_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = height_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensity_map[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    # RGB_Map = torch.unsqueeze(RGB_Map, 0)
    # print(RGB_Map.shape)
    return RGB_Map.numpy()

# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)
