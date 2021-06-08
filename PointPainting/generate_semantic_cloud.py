import argparse, cv2, time, tqdm, sys
from PIL import Image
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import pathlib as Path
import torch
import pickle

sys.path.insert(0, '../')

from visualization.KittiUtils import *
from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer

from .BiseNetv2 import BiSeNetV2
from .pointpainting import PointPainter
from .paint_utils import postprocessing, preprocessing_kitti
from .label import trainId2label

from Models.SFA.data_process.kitti_bev_utils import makeBEVMap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def bev_to_colored_bev_semantic(bev):
    semantic_map = bev[:,:,3]
    shape = semantic_map.shape[:2]
    color_map = np.zeros((shape[0], shape[1], 3))

    for id in trainId2label:
        label = trainId2label[id]
        if id == 255 or id == -1:
            continue
        color = label.color
        color_map[semantic_map == id] = color[2], color[1], color[0]
    return color_map

def main(args):

    # Semantic Segmentation
    bisenetv2 = BiSeNetV2()
    torch.cuda.empty_cache()
    checkpoint = torch.load(args.weights_path, map_location=device)
    bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
    bisenetv2.eval()
    bisenetv2.to(device)

    # Fusion
    painter = PointPainter()

    # Visualizer
    visualizer = KittiVisualizer()

    dataset_root = os.path.join(args.data_path, "training")
    KITTI = KittiDataset(dataset_root, mode='all')

    for i in range(len(KITTI)):
        image, pointcloud, _, calib = KITTI[i]

        input_image = preprocessing_kitti(image)
        semantic = bisenetv2(input_image)
        semantic = postprocessing(semantic)

        painted_pointcloud = painter.paint(pointcloud, semantic, calib)
        bev = makeBEVMap(painted_pointcloud, None, pointpainting=True)

        if args.vis:
            # scene_2D = visualizer.bev_to_colored_bev_semantic(bev)
            bev = bev_to_colored_bev_semantic(bev)
            cv2.imshow("scene", bev)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_path', type=str, default='../checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--data_path', type=str, default='../data/kitti')
    parser.add_argument('--save_path', type=str, default='../data/kitti/training/velodyne_semantic',)
    parser.add_argument('--vis', action='store_true', default=True, help='visualize')

    args = parser.parse_args()
    main(args)

