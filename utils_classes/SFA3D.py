import argparse
import sys
import os
import time
import warnings
import numpy as np
from pathlib import Path

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

from Models.SFA.utils.misc import make_folder, time_synchronized
from Models.SFA.utils.evaluation_utils import decode, post_processing, convert_det_to_real_values
from Models.SFA.utils.torch_utils import _sigmoid

import Models.SFA.models.fpn_resnet as fpn_resnet


class SFA3D:
    def __init__(self, configs):
        self.configs = configs
        self.model = self.load_model()

    def load_model(self):
        # get layers num
        arch_parts = self.configs.arch.split('_')
        num_layers = int(arch_parts[-1])
        # load model
        model = fpn_resnet.get_pose_net(num_layers=num_layers,
                                        heads=self.configs.heads, 
                                        head_conv=self.configs.head_conv,
                                        imagenet_pretrained=self.configs.imagenet_pretrained
                                        )
        # load weights
        model.load_state_dict(torch.load(self.configs.pretrained_sfa, map_location='cpu'))
        print('Loaded weights from {}\n'.format(self.configs.pretrained_sfa))
        self.configs.device = torch.device('cpu' if self.configs.no_cuda else 'cuda:0')

        # convert to cuda & eval mode
        model = model.to(device=self.configs.device)
        model.eval()
        return model
    
    def predict(self, bev, printer=False):
        # convert to bird eye view -> get heatmap output -> convert to kitti format output
        if printer:
            start = time_synchronized()
            outputs = self.model(bev)
            detections = self.post_procesiing(outputs)
            end = time_synchronized()
            print(f"Time for SFA: {1000 * (end - start)} ms\n")
        else:
            outputs = self.model(bev)
            detections = self.post_procesiing(outputs)
        return detections

    def post_procesiing(self, outputs):
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
        # detections size (batch_size, K, 10)
        detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],outputs['dim'], K=self.configs.K)
        detections = detections.cpu().detach().numpy().astype(np.float32)
        detections = post_processing(detections, self.configs.num_classes, self.configs.down_ratio, self.configs.peak_thresh)
        detections = detections[0]
        detections = convert_det_to_real_values(detections)
        return detections
    