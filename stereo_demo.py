import argparse, cv2
import torch.backends.cudnn as cudnn
from full_demo import parse_config
from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation

def preprocess_visualization(disparity):

    disparity = disparity.cpu().squeeze().numpy()

    # standerlization .. convert it to 0-1 range
    min_disp = np.min(disparity)
    max_disp = np.max(disparity)
    disparity = (disparity - min_disp) / (max_disp - min_disp)
    # convert 0-1 range - 0-255 range
    disparity = (disparity*256).astype(np.uint16)

    # np.set_printoptions(threshold=np.inf)
    # print(disparity)

    return disparity

def main():
    args, cfg = parse_config()
    cudnn.benchmark = True

    KITTI = KittiDataset(args.datapath, stereo_mode=True)

    stereo_model = Stereo_Depth_Estimation(args, cfg)

    visualizer = KittiVisualizer()

    for i in range(args.index, len(KITTI)):
        imgL, imgR, labels, calib = KITTI[i]

        disparity, pointcloud = stereo_model.predict(imgL, imgR, calib.calib_path, return_disparity=True)
        disparity = preprocess_visualization(disparity)

        visualizer.visualize_stereo_scene(imgL, disparity, pointcloud)
        if visualizer.user_press == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
