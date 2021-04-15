import argparse, cv2, time
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import pathlib as Path
import cv2

from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
import visualization.BEVutils as BEVutils

from utils_classes.SFA3D import SFA3D
from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation

from sfa_demo import parse_test_configs, parse_config

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

visualizer = KittiVisualizer()

def compute_intersection_polygons(vertices1, vertices2):
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    return poly1.intersection(poly2).area
    
def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def box3d_iou(bbox1:BBox3D, bbox2:BBox3D, calib:KittiCalibration):
    ''' Compute 3D bounding box IoU.
    Input:
        bbox1 : KittiObject BBox3D in any coord (will be converted to LIDAR coord.)
        bbox2 : KittiObject BBox3D in any coord (will be converted to LIDAR coord.)
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    global visualizer

    # get corners in LIDAR coord. (front x, left y, up z)
    corners1 = bbox3d_to_corners(bbox1, calib)
    corners2 = bbox3d_to_corners(bbox2, calib)
    
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in range(4)]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in range(4)]

    # permutate to be counter clockwise
    rect1[2], rect1[3] = rect1[3], rect1[2]
    rect2[2], rect2[3] = rect2[3], rect2[2]

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter_area = compute_intersection_polygons(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)

    zmax = min(corners1[0,2], corners2[0,2])
    zmin = max(corners1[4,2], corners2[4,2])

    inter_vol = inter_area * max(0.0, zmax-zmin)
    
    vol1 = bbox1.volume()
    vol2 = bbox2.volume()
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def bbox3d_to_corners(bbox3d, calib):
    global visualizer
    corners = visualizer.convert_3d_bbox_to_corners(bbox3d, calib)
    return corners


def max_iou_3D(object_iou:KittiObject, objects:list, calib):
    """ 
        Compute IOU 3D of givin bbox for all labels bboxes and get the max of them
        Args:
            object_iou: predicted KittiObject with bbox in LIDAR coord.
            objects: list of KittiObject to calculate the IOU with all of them 
        Return:
            Maximum IOU of bbox wrt the labels bboxes
    """
    max_iou_3d = 0
    max_iou_bev = 0
    bbox_iou = object_iou.bbox_3d 
    for kitti_object in objects:
        # skip other classes
        if kitti_object.label != object_iou.label:
            continue

        bbox = kitti_object.bbox_3d
        iou_3d, iou_bev = box3d_iou(bbox_iou, bbox, calib)
        if iou_3d > max_iou_3d:
            max_iou_3d = iou_3d
            max_iou_bev = iou_bev

    return max_iou_3d, max_iou_bev

    
def main():
    cfg, args = parse_test_configs()
    stereo_args = parse_config()
    cudnn.benchmark = True
    global visualizer

    dataset_root = os.path.join(cfg.dataset_dir, "training")
    KITTI = KittiDataset(dataset_root, mode='train')
    sfa_model = SFA3D(cfg) 

    # KITTI_stereo = KittiDataset(dataset_root, stereo_mode=True)    
    # anynet_model = Stereo_Depth_Estimation(stereo_args,None)

    # ======================================================================
    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]
        detections = sfa_model.predict(pointcloud)
        objects = SFA3D_output_to_kitti_objects(detections)

        # IOU 
        for obj in objects:
            iou_3d, iou_bev = max_iou_3D(obj, labels, calib)
            print(f'{label_to_class_name(obj.label)} iou 3d({iou_3d}) iou bev({iou_bev})')
            print('='*30)

        # visualizer.visualize_scene_3D(pointcloud, objects, labels, calib)
        visualizer.visualize_scene_2D(pointcloud, image, objects, labels, calib)
        if visualizer.user_press == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

