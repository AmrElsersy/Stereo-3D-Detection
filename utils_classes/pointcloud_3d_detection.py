import torch
import torch.nn.parallel
import torch.utils.data
import time
from pathlib import Path
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from utils_classes.pointcloud_preprocessing import PointcloudPreprocessing

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