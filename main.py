import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
import time
import glob
from pathlib import Path
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
from preprocessing.generate_lidar import project_disp_to_points, project_depth_to_points, Calibration
from dataloader import diy_dataset as ls

import models.anynet
import mayavi.mlab as mlab

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V

from configrations import *

from visualization.KittiDataset import *
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *

def parse_config():
    pvrcnn = PVRCNN()
    pointpillars = PointPillars()
    paper = pointpillars

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
    parser.add_argument('--pretrained', type=str, default='checkpoint/kitti2015_ck/checkpoint.tar',help='pretrained model path')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--cfg_file', type=str, default=paper.cfg,help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=Path.joinpath(Path.home(), "Stereo-3D-Detection/path-to-kitti") )
    parser.add_argument('--ckpt', type=str, default=paper.model, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    cudnn.benchmark = True

    test_left_img, test_right_img, test_left_disp = ls.testloader(args.datapath)

    stereoLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)
    
    stereo_model = Stereo_Depth_Estimation(args, cfg)
    pointpillars = PointCloud_3D_Detection(args, cfg)

    visualizer = KittiVisualizer()

    for imgL, imgR, _ in stereoLoader:

        psuedo_pointcloud = stereo_model.predict(imgL, imgR)
        pred = pointpillars.predict(psuedo_pointcloud)

        objects = model_output_to_kitti_objects(pred)
        # visualizer.visualize(psuedo_pointcloud, objects)      
        visualizer.visualize_bev(psuedo_pointcloud, objects)
        visualizer.show()
  



class PointcloudPreprocessing(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
    def preprocess_pointcloud(self, pointcloud):
        input_dict = {
            'points': pointcloud,
            'frame_id': 0
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class PointCloud_3D_Detection:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.logger = common_utils.create_logger()

        self.preprocesiing = PointcloudPreprocessing(
            dataset_cfg=self.cfg.DATA_CONFIG, class_names=self.cfg.CLASS_NAMES, training=False, root_path=Path(self.args.data_path), logger=self.logger
        )        

        self.model = self.load_model()

    def load_model(self):
        model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.preprocesiing)
        model.load_params_from_file(filename=self.args.ckpt, logger=self.logger, to_cpu=True)
        model.cuda()
        model.eval()
        return model

    def predict(self, pointcloud):

        data_dict = self.preprocesiing.preprocess_pointcloud(pointcloud)

        with torch.no_grad():
            data_dict = self.preprocesiing.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            t1 = time.time()
            pred_dicts, _ = self.model.forward(data_dict)
            print(pred_dicts)
            t2 = time.time()
            print("3D Model time= ", t2 - t1)

            return pred_dicts

# ========================= Stereo =========================
class Stereo_Depth_Estimation:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.model = self.load_model()

    def load_model(self):
        model = models.anynet.AnyNet(self.args)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(self.args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    def predict(self, imgL, imgR):
        disparity = self.stereo_to_disparity(imgL, imgR)
        # print("disparity.shape", len(disparity))

        index = range(4) # [1,2,3,4]
        # get the last disparity (best accuracy)
        last_index = len(disparity) - 1
        output = disparity[-1]

        output = torch.squeeze(output,1)
        
        predix = str(index[0]).zfill(6)

        img_cpu = np.asarray(output.cpu())
        disp_map = np.clip(img_cpu[0, :, :], 0, 2**16)

        calib_file = '{}/{}.txt'.format(self.args.datapath + '/calib', predix)
        calib = Calibration(calib_file)
        
        disp_map = (disp_map*256).astype(np.uint16)/256.
        lidar = project_disp_to_points(calib, disp_map, self.args.max_high)

        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)

        return lidar

    def stereo_to_disparity(self, imgL, imgR):
        
        self.model.eval()

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        
        with torch.no_grad():
            startTime = time.time()
            outputs, all_time = self.model(imgL, imgR)
            all_time = ''.join(['At Stage {}: time {:.2f} ms ~ {:.2f} FPS\n'.format(
                x, (all_time[x]-startTime) * 1000,  1 / ((all_time[x]-startTime))) for x in range(len(all_time))])
            return outputs



if __name__ == '__main__':
    main()
