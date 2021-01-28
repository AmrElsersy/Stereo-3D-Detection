import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import time
from AnyNet.preprocessing.generate_lidar import project_disp_to_points, Calibration
from AnyNet.models.anynet import AnyNet
from visualization.KittiUtils import *
from utils_classes.stereo_preprocessing import StereoPreprocessing

# ========================= Stereo =========================
class Stereo_Depth_Estimation:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.model = self.load_model()
        self.preprocesiing = StereoPreprocessing()

    def load_model(self):
        model = AnyNet(self.args)
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
        output = torch.squeeze(output ,1)

        img_cpu = np.asarray(output.cpu())
        disp_map = np.clip(img_cpu[0, :, :], 0, 2** 16)

        calib = Calibration(calib_path)

        disp_map = (disp_map * 256).astype(np.uint16) / 256.
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
                x, (all_time[x] - startTime) * 1000, 1 / ((all_time[x] - startTime))) for x in range(len(all_time))])
            return outputs
