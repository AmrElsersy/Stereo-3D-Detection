import argparse
import torch.backends.cudnn as cudnn
from configs.configrations import *
from pcdet.config import cfg, cfg_from_yaml_file

from visualization.KittiDataset import KittiVideo
from visualization.KittiUtils import *
from visualization.KittiVisualization import KittiVisualizer
from utils_classes.pointcloud_3d_detection import PointCloud_3D_Detection
import time
import cv2

pointpillars = PointPillars()
paper = pointpillars

def parse_config():
    parser = argparse.ArgumentParser(description='KITTI Demo Video')
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

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():

    args, cfg = parse_config()
    cudnn.benchmark = True
    pointpillars = PointCloud_3D_Detection(args, cfg)
    visualizer = KittiVisualizer()

    # KITTI Video
    VIDEO_ROOT_PATH = '/home/ayman/FOE-Linux/Graduation_Project/KITTI/2011_09_26_drive_0001'

    dataset = KittiVideo(
            img_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_drive_0001_sync/2011_09_26/image_02/data"),
            lidar_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_drive_0001_sync/2011_09_26/velodyne_points/data"),
            calib_dir=os.path.join(VIDEO_ROOT_PATH, "2011_09_26_calib/2011_09_26")
        )


    img_list = []
    avg_time = 0.
    for i in range(len(dataset)-80):
        imgL, pointcloud, calib = dataset[i]
        # Prediction
        t = time.time()
        pred = pointpillars.predict(pointcloud)
        avg_time += (time.time() - t)

        objects = model_output_to_kitti_objects(pred)
        img_ = visualizer.visualize_scene_image(imgL, objects, calib)
        img_list.append(img_)

    height, width, channels = dataset[0][0].shape
    avg_time = avg_time / len(dataset)
    FPS = 1 / avg_time     
    print("Samples Average Time",avg_time)
    print("FPS", FPS)
    outVideo = cv2.VideoWriter('demo_video_.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (width, height))
    
    for img in img_list:
        outVideo.write(img)

if __name__ == '__main__':
    main()

