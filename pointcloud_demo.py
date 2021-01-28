import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.visual_utils import visualize_utils as V

from time import time
from configs.configrations import *

pvrcnn = PVRCNN()
pointpillars = PointPillars()
second = Second()
pointrcnn = PointRCNN()
pointrcnn_iou = PointRCNNIoU()
partfree = PartFree()
partanchor = PartAnchor()
paper = pointpillars


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=paper.cfg,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=paper.model, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    avg_time = 0
    avg_fps = 0
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: {idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            start = time()
            pred_dicts, _ = model.forward(data_dict)
            end = time()
            time_forward = (end - start) * 1000
            if idx > 0:
                avg_time += time_forward
                avg_fps += 1000 / time_forward
            logger.info(f'Time in ms for sample index: {idx + 1} is {"{:.2f}".format(time_forward)} ms')
            logger.info(f'FPS for sample index: {idx + 1} is {"{:.2f}".format(1000 / time_forward)} frame per second')

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    avg_time /= len(demo_dataset) - 1
    avg_fps /= len(demo_dataset) - 1
    logger.info(f'Average Time in ms for {paper.type}: is {"{:.2f}".format(avg_time)} ms')
    logger.info(f'Average FPS for {paper.type}: is {"{:.2f}".format(avg_fps)} frame per second')
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
