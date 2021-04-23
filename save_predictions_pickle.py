import pickle
import argparse

from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
import visualization.BEVutils as BEVutils

from utils_classes.SFA3D import SFA3D
from sfa_demo import parse_test_configs, parse_config
from full_demo import parse_config_pillars


def save_predictions():
    cudnn.benchmark = True
    global visualizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sfa', 'pointpillars'], default='sfa', help='choose model to save its outputs')
    args_main = parser.parse_args()
    
    # dataset
    dataset_root = os.path.join(cfg.dataset_dir, "training")
    KITTI = KittiDataset(dataset_root, mode='val')

    # model
    model = None
    if args_main.model == 'sfa':
        cfg, args = parse_test_configs()
        model = SFA3D(cfg)
    else:
        args_pillars, cfg_pillars = parse_config_pillars()
        model = PointCloud_3D_Detection(args_pillars, cfg_pillars)

    # pickle save list
    objects_val = []

    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]

        pred = model.predict(pointcloud)

        # labels
        objects = None
        if args.model == 'sfa':
            objects = SFA3D_output_to_kitti_objects(pred)
        else:
            objects = model_output_to_kitti_objects(pred)

        objects_val.append(objects)

    with open('objects.pickle', 'wb') as f:
        pickle.dump(objects_val, f)

if __name__ == '__main__':
    save_predictions()
