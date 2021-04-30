import pickle
import argparse
import torch.backends.cudnn as cudnn
import torch

from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
import visualization.BEVutils as BEVutils

from utils_classes.SFA3D import SFA3D
from sfa_demo import parse_test_configs 
# from sfa_demo import parse_config
# from full_demo import parse_config_pillars
torch.cuda.empty_cache()

def save_predictions():
    cudnn.benchmark = True
    visualizer = KittiVisualizer()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sfa', 'pillars'], default='sfa', help='choose model to save its outputs')
    args_main = parser.parse_args()
    
    # dataset
    dataset_root = os.path.join('data', 'kitti', "training")
    KITTI = KittiDataset(dataset_root, mode='val')

    # model
    model = None
    if args_main.model == 'sfa':
        cfg, args = parse_test_configs(parser)
        model = SFA3D(cfg)
    else:
        args_pillars, cfg_pillars = parse_config_pillars()
        model = PointCloud_3D_Detection(args_pillars, cfg_pillars)

    # pickle save list
    predictions = []

    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]

        # predictions
        torch.cuda.empty_cache()
        pred = model.predict(pointcloud)

        # kitti objects
        objects = None
        if args_main.model == 'sfa':
            objects = SFA3D_output_to_kitti_objects(pred)
        else:
            objects = model_output_to_kitti_objects(pred)

        predictions.append(objects)
        print(i)
        # visualizer.visualize_scene_2D(pointcloud, image, objects, labels, calib=calib)
        # if visualizer.user_press == 27:
        #     cv2.destroyAllWindows()
        #     break

    with open('predictions_' + args_main.model + '.pickle', 'wb') as f:
        pickle.dump(predictions, f)

if __name__ == '__main__':
    save_predictions()
