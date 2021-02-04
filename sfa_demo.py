import argparse, cv2, time
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import pathlib as Path

from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *

from utils_classes.SFA3D import SFA3D

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN', help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH', help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default='SFA3D/checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_30.pth', metavar='PATH')
    parser.add_argument('--K', type=int, default=50, help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None, help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true', help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--SDN', action='store_true', help="Psuedo LIDAR from SDN Depth Estimation")
    parser.add_argument('--index', type=int, default=0, help="start index in dataset")
    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    # #### set it to empty as this file is inside the root of the project ####
    configs.root_dir = ''
    configs.dataset_dir = os.path.join(configs.root_dir, 'data', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    args = parser.parse_args()
    
    return configs, args


def main():
    cfg, args = parse_test_configs()
    cudnn.benchmark = True

    KITTI = KittiDataset(os.path.join(cfg.dataset_dir, "testing"))
    sfa_model = SFA3D(cfg) 
    visualizer = KittiVisualizer()

    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]
        detections = sfa_model.predict(pointcloud)
        print(detections)

        time.sleep(2)
        # objects = SFA3D_output_to_kitti_objects(detections)

        # visualizer.visualize_scene_2D(pointcloud, image, objects, calib=calib)
        # if visualizer.user_press == 27:
        #     cv2.destroyAllWindows()
        #     break

if __name__ == '__main__':
    main()


if __name__ == '__ray2__':
    configs, args = parse_test_configs()


    # ============================= Sersy Edit =======================================
    stereo_model = Stereo_Depth_Estimation(stereo_args,None)

    dataset_root = os.path.join(configs.dataset_dir, 'testing')
    print(configs.dataset_dir)
    KITTI_stereo = KittiDataset(dataset_root, stereo_mode=True)    
    KITTI = KittiDataset(dataset_root, stereo_mode=False)   

    winname = "image"

    # SDN pseudo
    if args.SDN:
        n = len(KITTI)
        for i in range(int(n/2), n):
            image, pointcloud, labels, calib = KITTI[i]

            complex_yolo(pointcloud)
        
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 800,300)
            cv2.imshow(winname, image)
            if cv2.waitKey(0) == 27:
                break
            cv2.destroyAllWindows()

    # Stereo psuedo
    else:        
        for i in range(len(KITTI_stereo)):
            imgL, imgR, _, calib = KITTI_stereo[i]

            pointcloud = stereo_model.predict(imgL, imgR, calib.calib_path)
            complex_yolo(pointcloud)

            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 800,300)
            cv2.imshow(winname, imgL)
            if cv2.waitKey(0) == 27:
                break
            cv2.destroyAllWindows()

