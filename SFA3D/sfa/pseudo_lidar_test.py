"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from PIL import Image

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import get_filtered_lidar

from data_process.kitti_bev_utils import makeBEVMap
import config.kitti_config as cnf
from pathlib import Path

# ========================= Stereo =========================
root = '../../'
sys.path.insert(1, root)

from AnyNet.dataloader import KITTILoader as DA
from AnyNet.dataloader import preprocess
from AnyNet.models import anynet

from AnyNet.preprocessing.generate_lidar import project_disp_to_points, project_depth_to_points
from AnyNet.preprocessing.generate_lidar import Calibration

from visualization.KittiDataset import KittiDataset
from visualization.KittiUtils import *

from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation

def parse_config():

    parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
    parser.add_argument('--maxdisp', type=int, default=192,help='maxium disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--max_disparity', type=int, default=192)
    parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
    parser.add_argument('--datatype', default='2015',help='datapath')
    parser.add_argument('--datapath', default=None, help='datapath')
    parser.add_argument('--epochs', type=int, default=300,help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=6,help='batch size for training (default: 6)')
    parser.add_argument('--test_bsize', type=int, default=8,help='batch size for testing (default: 8)')
    parser.add_argument('--save_path', type=str, default='results/pseudoLidar/',help='the path of saving checkpoints and log')
    parser.add_argument('--resume', type=str, default=None,help='resume path')
    parser.add_argument('--lr', type=float, default=5e-4,help='learning rate')
    parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
    parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
    parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
    parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
    parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
    parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
    parser.add_argument('--start_epoch_for_spn', type=int, default=121)
    parser.add_argument('--pretrained', type=str, default=os.path.join(root, 'configs/checkpoint/kitti2015_ck/checkpoint.tar'),help='pretrained model path')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=Path.joinpath(Path.home(), "Stereo-3D-Detection/path-to-kitti") )
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--SDN', action='store_true', help="Psuedo LIDAR from SDN Depth Estimation")
    args = parser.parse_args()

    return args

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN', help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH', help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default='../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_30.pth', metavar='PATH')
    parser.add_argument('--K', type=int, default=50, help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None, help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true', help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH', help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18',metavar='PATH',help='video path if the output format is video')
    parser.add_argument('--output-width', type=int, default=608, help='the width of showing output, the height maybe vary')
    parser.add_argument('--SDN', action='store_true', help="Psuedo LIDAR from SDN Depth Estimation")
    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, '../' , 'data', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    args = parser.parse_args()
    
    return configs, args


def complex_yolo(pointcloud):
    pointcloud = get_filtered_lidar(pointcloud, cnf.boundary)
    print(pointcloud[:,3])
    bev_maps = makeBEVMap(pointcloud, cnf.boundary)
    bev_maps = torch.from_numpy(bev_maps)
    bev_maps = torch.unsqueeze(bev_maps, 0)

    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    t1 = time_synchronized()
    outputs = model(input_bev_maps)
    outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
    outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
    # detections size (batch_size, K, 10)
    detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],outputs['dim'], K=configs.K)
    detections = detections.cpu().detach().numpy().astype(np.float32)
    detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
    t2 = time_synchronized()
    detections = detections[0]  # only first batch
    # Draw prediction in the image
    bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)    
    bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
    bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    cv2.imshow("BEV", bev_map)
    print('\tDone testing in time: {:.1f}ms, speed {:.2f}FPS'.format((t2 - t1) * 1000,1 / (t2 - t1)))
    
if __name__ == '__main__':
    stereo_args=  parse_config()
    configs, args = parse_test_configs()
    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:0')
    model = model.to(device=configs.device)
    model.eval()

    # ============================= Sersy Edit =======================================
    stereo_model = Stereo_Depth_Estimation(stereo_args,None)

    dataset_root = os.path.join(configs.dataset_dir, 'testing')
    print(configs.dataset_dir)
    KITTI_stereo = KittiDataset(dataset_root, stereo_mode=True)    
    KITTI = KittiDataset(dataset_root, stereo_mode=False)   

    winname = "image"

    # SDN pseudo
    if args.SDN:
        n = len(KITTI)
        for i in range(int(n/2), n):
            image, pointcloud, labels, calib = KITTI[i]

            complex_yolo(pointcloud)
        
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 800,300)
            cv2.imshow(winname, image)
            if cv2.waitKey(0) == 27:
                break
            cv2.destroyAllWindows()

    # Stereo psuedo
    else:        
        for i in range(len(KITTI_stereo)):
            imgL, imgR, _, calib = KITTI_stereo[i]

            pointcloud = stereo_model.predict(imgL, imgR, calib.calib_path)
            complex_yolo(pointcloud)

            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 800,300)
            cv2.imshow(winname, imgL)
            if cv2.waitKey(0) == 27:
                break
            cv2.destroyAllWindows()


