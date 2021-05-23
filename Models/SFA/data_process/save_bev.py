import sys
import os
import math
from builtins import int

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch


from .kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from .kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
from . import transformation
from ..config import kitti_config as cnf
import random
from .transformation import OneOf, Random_Rotation, Random_Scaling

from .kitti_dataset import KittiDataset
from ..config.train_config import parse_train_configs

torch.cuda.empty_cache()

def get_dataset(configs):
    train_lidar_aug = OneOf([
        Random_Rotation(limit_angle=np.pi / 4, p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0),
        ], p=0.66)
    train_dataset = KittiDataset(configs, mode='train', lidar_aug=train_lidar_aug, hflip_prob=configs.hflip_prob,
                                 num_samples=configs.num_samples, bev_saving=True)
    return train_dataset

def get_configs():
    configs = parse_train_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))
    return configs


def save_bev_tensors(dataset):
    start = 3000
    samples = 6000
    for i in range(start, samples):
        path, bev = dataset.save_bev(i)

        if i % 10 == 0:
            print(path)

def test(dataset):
    print('test')

if __name__ == '__main__':
    configs = get_configs()
    dataset = get_dataset(configs)

    save_bev_tensors(dataset)
    test(dataset)
