import argparse
import torch.backends.cudnn as cudnn
from configs.configrations import *
from pcdet.config import cfg, cfg_from_yaml_file
from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation
from utils_classes.pointcloud_3d_detection import PointCloud_3D_Detection
from utils_classes.pointcloud2detection import predict_lidar, predict_pseudo_lidar

pvrcnn = PVRCNN()
pointpillars = PointPillars()
second = Second()
pointrcnn = PointRCNN()
pointrcnn_iou = PointRCNNIoU()
partfree = PartFree()
partanchor = PartAnchor()
paper = pointpillars


def parse_config():

    parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
    parser.add_argument('--maxdisp', type=int, default=192,help='maxium disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--max_disparity', type=int, default=192)
    parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
    parser.add_argument('--datatype', default='2015',help='datapath')
    parser.add_argument('--datapath', default='data/kitti/training', help='datapath')
    parser.add_argument('--epochs', type=int, default=300,help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=6,help='batch size for training (default: 6)')
    parser.add_argument('--test_bsize', type=int, default=8,help='batch size for testing (default: 8)')
    parser.add_argument('--save_path', type=str, default='results/pseudoLidar/',help='the path of saving checkpoints and log')
    parser.add_argument('--resume', type=str, default=None,help='resume path')
    parser.add_argument('--lr', type=float, default=5e-4,help='learning rate')
    parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
    parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
    parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
    parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
    parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
    parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
    parser.add_argument('--start_epoch_for_spn', type=int, default=121)
    parser.add_argument('--pretrained', type=str, default='configs/checkpoint/kitti2015_ck/checkpoint.tar',help='pretrained model path')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--cfg_file', type=str, default=paper.cfg,help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training')
    parser.add_argument('--ckpt', type=str, default=paper.model, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--lidar_only', action='store_true')
    parser.add_argument('--psuedo', action='store_true')
    parser.add_argument('--index', type=int, default=0, help='index of an example in the dataset')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    cudnn.benchmark = True

    KITTI = KittiDataset('data/kitti/training', stereo_mode=True)

    stereo_model = Stereo_Depth_Estimation(args, cfg)
    pointpillars = PointCloud_3D_Detection(args, cfg)

    visualizer = KittiVisualizer()

    if args.lidar_only:
        predict_lidar(pointpillars)
        return
    if args.psuedo :
        predict_pseudo_lidar(pointpillars)
        return

    # for imgL, imgR, _ in stereoLoader:
    # for i in range(8,100):
    imgL, imgR, labels, calib_path = KITTI[args.index]
    calib = KittiCalibration(calib_path)

    psuedo_pointcloud = stereo_model.predict(imgL, imgR, calib_path)
    pred = pointpillars.predict(psuedo_pointcloud)
    objects = model_output_to_kitti_objects(pred)

    visualizer.visualize_scene_2D(psuedo_pointcloud, imgL, objects, calib=calib)
    # visualizer.visualize_scene_3D(psuedo_pointcloud, objects, labels, calib)
    # visualizer.visualize_scene_bev(psuedo_pointcloud, objects, calib=calib)
    # visualizer.visualize_scene_image(imgL, objects, calib)


if __name__ == '__main__':
    main()
