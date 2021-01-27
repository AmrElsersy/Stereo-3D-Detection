import argparse
import os, sys
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
from dataloader import preprocess
import utils.logger as logger
import torch.backends.cudnn as cudnn
from preprocessing.generate_lidar import project_disp_to_points, project_depth_to_points, Calibration
from dataloader import diy_dataset as ls

import models.anynet
import mayavi.mlab as mlab
from configrations import *

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

sys.path.insert(1, 'visualization')
from KittiDataset import *
from KittiVisualization import KittiVisualizer
from KittiUtils import *

def parse_config():
    # pvrcnn = PVRCNN()
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
    parser.add_argument('--lidar_only', action='store_true')
    parser.add_argument('--psuedo', action='store_true')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    cudnn.benchmark = True
    # test_left_img, test_right_img, test_left_disp = ls.testloader(args.data_path)
    # stereoLoader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    #     batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    KITTI = KittiDataset('/home/amrelsersy/SFA3D/dataset/kitti/testing', stereo_mode=True)
    # stereoLoader = torch.utils.data.DataLoader(KITTI, 1, shuffle=False, num_workers=4, drop_last=False)

    stereo_model = Stereo_Depth_Estimation(args, cfg)
    pointpillars = PointCloud_3D_Detection(args, cfg)

    visualizer = KittiVisualizer()

    if args.lidar_only:
        main_lidar(pointpillars)
        return
    if args.psuedo :
        main_pseudo(pointpillars)
        return

    # for imgL, imgR, _ in stereoLoader:
    for i in range(8,100):
        imgL, imgR, labels, calib_path = KITTI[i]
        calib = KittiCalibration(calib_path)

        psuedo_pointcloud = stereo_model.predict(imgL, imgR, calib_path)
        print(psuedo_pointcloud.shape)
        pred = pointpillars.predict(psuedo_pointcloud)

        objects = model_output_to_kitti_objects(pred)
        visualizer.visualize_scene_2D(psuedo_pointcloud, imgL, objects, calib=calib)
        # visualizer.visualize_scene_3D(psuedo_pointcloud, objects, labels, calib)      
        # visualizer.visualize_scene_bev(psuedo_pointcloud, objects, calib=calib)
        # visualizer.visualize_scene_image(imgL, objects, calib)

def main_pseudo(model):
    visualizer = KittiVisualizer()
    KITTI = KittiDataset("/home/amrelsersy/SFA3D/dataset/kitti/testing")
    root = "/home/amrelsersy/SFA3D/dataset/kitti/testing/pseudo_SDN"
    paths = os.listdir(root)

    for path in paths:
        path = root+"/"+path 
        pointcloud = KITTI.read_pointcloud_bin(path)
        image = KITTI.read_image_cv2(path.replace("pseudo_SDN","image_2").replace("bin", "png"))
        calib = KittiCalibration(path.replace("pseudo_SDN","calib").replace("bin", "txt"))
        print(pointcloud.shape, image.shape)

        pred = model.predict(pointcloud)
        objects = model_output_to_kitti_objects(pred)

        # cv2.imshow("image", image)      
        visualizer.visualize_scene_2D(pointcloud, image, objects, calib)
        # visualizer.visualize_scene_bev(pointcloud, objects)
        # visualizer.visualize_scene_3D(pointcloud, objects)
        # visualizer.visualize_scene_image(image, objects, calib)

def main_lidar(model):
    print("========== LIDAR only =============")
    visualizer = KittiVisualizer()
    KITTI = KittiDataset('/home/amrelsersy/SFA3D/dataset/kitti/testing')

    for i in range(8, 100):
        image, pointcloud, labels, calib = KITTI[i]
        print(pointcloud.shape)

        pred = model.predict(pointcloud)

        objects = model_output_to_kitti_objects(pred)
        # visualizer.visualize_scene_3D(pointcloud, objects, labels, calib)      
        visualizer.visualize_scene_bev(pointcloud, objects, calib=calib)
        # visualizer.visualize_scene_image(image, objects, calib)

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
        self.preprocesiing = StereoPreprocessing()

    def load_model(self):
        model = models.anynet.AnyNet(self.args)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(self.args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    def predict(self, imgL, imgR, calib_path):

        imgL, imgR = self.preprocesiing.preprocess(imgL, imgR)
        disparity = self.stereo_to_disparity(imgL, imgR)

        # get the last disparity (best accuracy)
        last_index = len(disparity) - 1
        output = disparity[-1]
        output = torch.squeeze(output,1)
        
        img_cpu = np.asarray(output.cpu())
        disp_map = np.clip(img_cpu[0, :, :], 0, 2**16)

        calib = Calibration(calib_path)
        
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

class StereoPreprocessing:
    def __init__(self, training=False):
        self.training = training
    def preprocess(self, left_img, right_img):
        # if type(left_img) == np.ndarray:
        left_img = Image.fromarray(np.uint8(left_img)).convert('RGB')
        right_img = Image.fromarray(np.uint8(right_img)).convert('RGB')
        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img
        else:
            w, h = left_img.size

            left_img = left_img.crop((w-1232, h-368, w, h))
            right_img = right_img.crop((w-1232, h-368, w, h))
            w1, h1 = left_img.size

            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)


            # convert [3, 368, 1232] to tensor [1, 3, 368, 1232]
            left_img  = left_img.clone().detach().reshape(1, *left_img.size()) 
            right_img = right_img.clone().detach().reshape(1, *right_img.size())

            return left_img, right_img

if __name__ == '__main__':
    main()
