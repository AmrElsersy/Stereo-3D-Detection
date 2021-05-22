import argparse, cv2, time
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import pathlib as Path
import torch
import pickle


from visualization.KittiUtils import *
from visualization.KittiDataset import KittiDataset
from visualization.KittiDataset import KittiVideo
from visualization.KittiVisualization import KittiVisualizer

from utils_classes.SFA3D import SFA3D
from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation

torch.cuda.empty_cache()

def parse_configs(parser = None):
    if not parser:
        parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN', help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH', help='The name of the model architecture')
    parser.add_argument('--pretrained_sfa', type=str, default='checkpoints/sfa.pth', metavar='PATH')
    parser.add_argument('--K', type=int, default=50, help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int, help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None, help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true', help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--index', type=int, default=0, help="start index in dataset")
    parser.add_argument('--maxdisp', type=int, default=192,help='maxium disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--max_disparity', type=int, default=192)
    parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
    parser.add_argument('--datatype', default='2015',help='datapath')
    parser.add_argument('--datapath', default='../KITTI/training', help='datapath')
    parser.add_argument('--epochs', type=int, default=300,help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=6,help='batch size for training (default: 6)')
    parser.add_argument('--test_bsize', type=int, default=8,help='batch size for testing (default: 8)')
    parser.add_argument('--resume', type=str, default=None,help='resume path')
    parser.add_argument('--lr', type=float, default=5e-4,help='learning rate')
    parser.add_argument('--with_spn', action='store_true', default=True, help='with spn network or not')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
    parser.add_argument('--init_channels', type=int, default=1, help='in    itial channels for 2d feature extractor')
    parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
    parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
    parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
    parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
    parser.add_argument('--start_epoch_for_spn', type=int, default=121)
    parser.add_argument('--pretrained_anynet', type=str, default='checkpoints/anynet.tar',help='pretrained model path')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--pseudo', action='store_true')

    parser.add_argument('--generate_pickle', action='store_true', help='If true, generate pickle file.')
    parser.add_argument('--generate_video', action='store_true', help='If true, generate video.')
    parser.add_argument('--save_path', type=str, default='results/',help='the path of saving video and pickle files')
    args = parser.parse_args()
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
    
    args = parser.parse_args()
    
    return configs, args

def main():
    cfg, args = parse_configs()
    cudnn.benchmark = True

    sfa_model = SFA3D(cfg)
    anynet_model = Stereo_Depth_Estimation(args, cfg)

    visualizer = KittiVisualizer()

    predictions = []

    if args.generate_video:
        img_list = []
        VIDEO_ROOT_PATH = 'data/demo'
        dataset = KittiVideo(
                imgL_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_0001/image_02/data"),
                imgR_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_0001/image_03/data"),
                lidar_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_0001/velodyne_points/data"),
                calib_dir=os.path.join(VIDEO_ROOT_PATH, "calib/2011_09_26")
            )
        loop_length=len(dataset)
    else:
        dataset_root = os.path.join(cfg.dataset_dir, "training")
        KITTI_stereo = KittiDataset(dataset_root, stereo_mode=True, mode='val')
        loop_length = len(KITTI_stereo)
    
    
    for i in range(args.index, loop_length):
        torch.cuda.empty_cache()
        if args.generate_video:
            imgL, imgR, pointcloud, calib = dataset[i]
        else:
            imgL, imgR, labels, calib = KITTI_stereo[i]

        BEV = anynet_model.predict(imgL, imgR, calib.calib_path)
        detections = sfa_model.predict(BEV)
        objects = SFA3D_output_to_kitti_objects(detections)

        if args.generate_pickle:
            predictions.append(objects)
            if i % 200 == 0:
                print(i)
        elif args.generate_video:
            img_ = visualizer.visualize_scene_image(imgL, objects, calib)
            img_list.append(img_)
        else:
            visualizer.visualize_scene_image(imgL, objects, calib=calib, scene_2D_mode=False)
            if visualizer.user_press == 27:
                cv2.destroyAllWindows()
                break
    
    
    if args.generate_pickle:
        with open(args.save_path + '/sfa.pickle', 'wb') as f:
            pickle.dump(predictions, f)

    elif args.generate_video:
        height, width, channels = dataset[0][0].shape
        outVideo = cv2.VideoWriter(args.save_path + '/end-to-end_demo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (width, height))
        for img in img_list:
            outVideo.write(img)

if __name__ == '__main__':
    main()

