import argparse, cv2
import torch.backends.cudnn as cudnn
from full_demo import parse_config_pillars
from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
from utils_classes.pointcloud_3d_detection import PointCloud_3D_Detection

def main():
    args, cfg = parse_config_pillars()
    cudnn.benchmark = True

    KITTI = KittiDataset(args.datapath)
    pointpillars = PointCloud_3D_Detection(args, cfg)
    visualizer = KittiVisualizer()

    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]
        pred = pointpillars.predict(pointcloud)
        objects = model_output_to_kitti_objects(pred)

        visualizer.visualize_scene_2D(pointcloud, image, objects, calib=calib)
        if visualizer.user_press == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
