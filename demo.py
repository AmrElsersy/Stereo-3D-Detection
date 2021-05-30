import argparse, cv2, time
from PIL import Image
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import pathlib as Path
import torch
import pickle

from visualization.KittiUtils import *
from visualization.evaluation import *
from visualization.KittiDataset import KittiDataset
from visualization.KittiDataset import KittiVideo
from visualization.KittiVisualization import KittiVisualizer

from Models.SFA.utils.misc import time_synchronized

from utils_classes.SFA3D import SFA3D
from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation

torch.cuda.empty_cache()

def parse_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    
    parser.add_argument('--save_path', type=str, default='results/',help='the path of saving video and pickle files')
    parser.add_argument('--pretrained_anynet', type=str, default='checkpoints/anynet.tar',help='pretrained model path')
    parser.add_argument('--pretrained_sfa', type=str, default='checkpoints/sfa.pth', metavar='PATH')
    parser.add_argument('--data', type=str, default='kitti')
    parser.add_argument('--eval', action='store_true', help='If true, evaluate your pipeline.')
    parser.add_argument('--generate_video', action='store_true', help='If true, generate video.')
    parser.add_argument('--with_bev', action='store_true', help='If true, generate video.')
    parser.add_argument('--profiling', action='store_true', help='put small limit for loop length')
    parser.add_argument('--with_spn', action='store_true', default=True, help='Allow using spn layer')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequence')

    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN', help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH', help='The name of the model architecture')
    parser.add_argument('--K', type=int, default=50, help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None, help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for loading data')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true', help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--index', type=int, default=0, help="start index in dataset")
    parser.add_argument('--maxdisp', type=int, default=192,help='maxium disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--max_disparity', type=int, default=192)
    parser.add_argument('--pseudo', action='store_true')

    """ Anynet modal args """
    parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
    parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
    parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
    parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
    parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
    parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
    """ LiDAR args """
    parser.add_argument('--max_high', type=int, default=1)

    """ Kitti sparsify args """
    parser.add_argument('--H', default=64, type=int)
    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=700, type=int)
    parser.add_argument('--slice', default=1, type=int)

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
    configs.dataset_dir = os.path.join('data', configs.data)
    # # An-Paths
    # configs.dataset_dir = '/home/ayman/FOE-Linux/Graduation_Project/KITTI'

        
    return configs

def main():
    cfg = parse_configs()
    cudnn.benchmark = True

    sfa_model = SFA3D(cfg)
    anynet_model = Stereo_Depth_Estimation(cfg)

    visualizer = KittiVisualizer()

    predictions = []

    if cfg.generate_video:
        img_list = []
        VIDEO_ROOT_PATH = 'data/' + 'demo'
        VIDEO_NAME = "2011_09_26_0106"
        # VIDEO_ROOT_PATH = '/home/ayman/FOE-Linux/Graduation_Project/KITTI/2011_09_26_drive_0001'
        dataset = KittiVideo(
                imgL_dir=os.path.join(VIDEO_ROOT_PATH, VIDEO_NAME + "/image_02/data"),
                imgR_dir=os.path.join(VIDEO_ROOT_PATH, VIDEO_NAME + "/image_03/data"),
                lidar_dir=os.path.join(VIDEO_ROOT_PATH, VIDEO_NAME + "/velodyne_points/data"),
                calib_dir=os.path.join(VIDEO_ROOT_PATH, "calib/2011_09_26")

                # An-Paths
                # imgL_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_drive_0001_sync/2011_09_26/image_02/data"),
                # imgR_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_drive_0001_sync/2011_09_26/image_03/data"),
                # lidar_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_drive_0001_sync/2011_09_26/velodyne_points/data"),
                # calib_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_calib/2011_09_26")

            )
        loop_length=len(dataset)
        avg_time = 0.
    else:
        dataset_root = os.path.join(cfg.dataset_dir, "training")
        KITTI_stereo = KittiDataset(dataset_root, stereo_mode=True, mode='val')
        loop_length = len(KITTI_stereo)
        if cfg.profiling:
            loop_length = 20

    for i in range(cfg.index, loop_length):
        if cfg.generate_video:
            imgL, imgR, pointcloud, calib = dataset[i]
            t = time_synchronized()
            printer = ((i % cfg.print_freq) == 0)
        else:
            imgL, imgR, labels, calib = KITTI_stereo[i]
            if cfg.eval:
                printer = False
            else:
                printer = True
        with torch.no_grad():
            if cfg.with_bev: 
                bev, sparse_points = anynet_model.predict(imgL, imgR, calib.calib_path, printer=printer)
            detections = sfa_model.predict(bev, printer=printer)
            objects = SFA3D_output_to_kitti_objects(detections)
        if cfg.eval:
            predictions.append(objects)
            if i % cfg.print_freq == 0:
                print(i)
        elif cfg.generate_video:
            if i == 0:
                continue 
            avg_time += (time_synchronized() - t)
            if cfg.with_bev:
                img_ = visualizer.visualize_scene_2D(sparse_points.cpu().numpy(), imgL, objects, calib=calib)
            else:
                img_ = visualizer.visualize_scene_image(imgL, objects, calib)
            img_list.append(img_)
        elif not cfg.profiling:
            visualizer.visualize_scene_image(imgL, objects, calib=calib, scene_2D_mode=False)
            if visualizer.user_press == 27:
                cv2.destroyAllWindows()
                break
        torch.cuda.empty_cache()

    if cfg.eval:
        evaluations = [
            Evaluation(iou_threshold=0.5, evaluate_class='Car',        mode=EvalMode.IOU_3D),
            Evaluation(iou_threshold=0.7, evaluate_class='Car',        mode=EvalMode.IOU_3D),
            Evaluation(iou_threshold=0.5, evaluate_class='Car',        mode=EvalMode.IOU_BEV),
            Evaluation(iou_threshold=0.7, evaluate_class='Car',        mode=EvalMode.IOU_BEV)
            # Evaluation(iou_threshold=0.5, evaluate_class='Pedestrian', mode=EvalMode.IOU_3D),
            # Evaluation(iou_threshold=0.5, evaluate_class='Pedestrian', mode=EvalMode.IOU_BEV),
            # Evaluation(iou_threshold=0.5, evaluate_class='Cyclist',    mode=EvalMode.IOU_3D),
            # Evaluation(iou_threshold=0.5, evaluate_class='Cyclist',    mode=EvalMode.IOU_BEV),
        ]

        mAP_strings = []
        for i in range(2):
            enable_filter = False
            if i == 1:
                enable_filter = True

            for evaluation in evaluations:
                mAP = evaluate(evaluation, predictions, enable_filter)
                mode = '3D' if evaluation.mode == EvalMode.IOU_3D else 'BEV'
                has_filter = 'filter = 0.3' if enable_filter else 'no filter'
                mAP_string = f'mAP = {str(mAP.item())}, {evaluation.evaluate_class_name}, IOU = {str(evaluation.iou_threshold)}, {mode}, {has_filter}'
                print(mAP_string)
                mAP_strings.append(mAP_string)
                
        file = open(os.path.join(cfg.save_path, 'demo_mAP.txt'), 'w')
        one_mAP_string = '\n'.join([string for string in mAP_strings])
        file.write(one_mAP_string)
        file.close()
        print('Saved @ File: ', cfg.save_path + '/demo_mAP.txt')

    elif cfg.generate_video:
        avg_time = avg_time / (loop_length-1)
        FPS = 1 / avg_time     
        print("Samples Average Time",avg_time)
        print("FPS", FPS)
        height, width, channels = img_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outVideo = cv2.VideoWriter(cfg.save_path + '/'+ VIDEO_NAME+'.mp4', fourcc, FPS, (width, height))
        for img in img_list:
            outVideo.write(img)

if __name__ == '__main__':
    main()

