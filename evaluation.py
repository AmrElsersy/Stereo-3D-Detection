import argparse, cv2, time
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import pathlib as Path
import cv2
import torch
import pickle

from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *
import visualization.BEVutils as BEVutils

from utils_classes.SFA3D import SFA3D
from utils_classes.stereo_depth_estimation import Stereo_Depth_Estimation
from sfa_demo import parse_test_configs, parse_config

from utils_classes.pointcloud_3d_detection import PointCloud_3D_Detection
from full_demo import parse_config_pillars

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

visualizer = KittiVisualizer()

torch.cuda.empty_cache()

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

    def bbox3d_to_corners(bbox3d, calib):
        global visualizer
        corners = visualizer.convert_3d_bbox_to_corners(bbox3d, calib)
        return corners

    def compute_intersection_polygons(vertices1, vertices2):
        poly1 = Polygon(vertices1)
        poly2 = Polygon(vertices2)
        return poly1.intersection(poly2).area
        
    def poly_area(x,y):
        """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

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

class EvalMode(Enum):
    IOU_3D = 0
    IOU_BEV = 1

class Evaluation:
    def __init__(self, iou_threshold, evaluate_class, mode):
        self.TP = []
        self.FP = []
        self.total_ground_truth = 0

        self.iou_threshold = iou_threshold
        self.evaluate_class = evaluate_class
        self.mode = mode

    def max_iou_3D(self, detection:KittiObject, labels:list, calib):
        """ 
            Compute IOU 3D of givin bbox for all labels bboxes and get the max of them
            Args:
                detection: predicted KittiObject with bbox in LIDAR coord.
                labels: list of KittiObject to calculate the IOU with all of them 
            Return:
                Maximum IOU of bbox wrt the labels bboxes
                    iou_3d: 3D IOU
                    iou_bev: BEV IOU
                    label_idx: index of the matched label
        """
        max_iou_3d = 0
        max_iou_bev = 0
        label_idx = None
        detected_bbox = detection.bbox_3d 

        for i, label in enumerate(labels):
            gt_bbox = label.bbox_3d
            iou_3d, iou_bev = box3d_iou(detected_bbox, gt_bbox, calib)

            if iou_3d > max_iou_3d:
                max_iou_3d = iou_3d
                max_iou_bev = iou_bev
                label_idx = i

        return max_iou_3d, max_iou_bev, label_idx

    def evaluate_step(self, detections:list, labels:list, calib:KittiCalibration):
        # filter classes
        for detection in detections:
            if detection.label != self.evaluate_class:
                detections.remove(detection)

        for label in labels:
            if label.label != self.evaluate_class:
                labels.remove(label)

        # [0, 0, 0] in case 3 predicted boxes
        TP = [0 for i in range(len(detections))]
        FP = [0 for i in range(len(detections))]
        total_gt = len(labels)

        # sort the detections to take the heighst score first to remove the gt boxes with heighst score box with IOU
        # we will remove gt boxes that is already been assigned to a detected box, so we want to have a heighst score detected box
        detections.sort(key=lambda kitti_obj: kitti_obj.score, reverse=True)

        # print(f'detections({len(detections)})', detections)
        # print(f'labels({len(labels)})', labels)
        # print('TP', TP)
        # print('FP', FP, '\n')

        # iou & update TP & FP
        for i, detection in enumerate(detections):

            iou_3d, iou_bev, label_idx = self.max_iou_3D(detection, labels, calib)
            iou = iou_3d if self.mode == EvalMode.IOU_3D else iou_bev
            
            if iou >= self.iou_threshold:
                TP[i] = 1
                # remove the gt_bbox (we cannot have multiple detections on the same gt_bbox and cound them all correct)
                del labels[label_idx]
            else:
                # if low iou or if not matched iou will be 0
                FP[i] = 1

            # print(f'({i}) det, ', detection)
            # print(f'labels({len(labels)})', labels)
            # print('TP', TP)
            # print('FP', FP, '\n')

        self.TP.extend(TP)
        self.FP.extend(FP)
        self.total_ground_truth += total_gt
        # print('mAPPPP',self.mAP())

    def mAP(self):
        """
            Compute mAP from the TP & FP lists of the evaluation class
        """
        TP = torch.tensor(self.TP)
        FP = torch.tensor(self.FP)
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # recall & precision
        epsilon = 1e-6
        recalls = TP_cumsum / (self.total_ground_truth + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # torch requirements to calculate area under curve (mAP)
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        # mAP = area under precision-recall curve
        average_precision = torch.trapz(precisions, recalls)
        return average_precision

def evaluate_pickle():
    cfg, args = parse_test_configs()
    cudnn.benchmark = True
    global visualizer

    dataset_root = os.path.join(cfg.dataset_dir, "training")
    KITTI = KittiDataset(dataset_root, mode='val')

    with open('predictions.pickle', 'rb') as f:
        preds_val = pickle.load(f)

    with open('objects.pickle', 'rb') as f:
        objects_val = pickle.load(f)

    evaluation = Evaluation(iou_threshold=0.5, evaluate_class=class_name_to_label('Car'), mode=EvalMode.IOU_3D)
    # ======================================================================
    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]
        objects = objects_val[i]

        # filter score
        for obj in objects:
            if obj.score < 0.5:
                objects.remove(obj)

        evaluation.evaluate_step(objects, labels, calib)
        if i % 20 == 0:
            print(f'{i}- mAP = {evaluation.mAP()}')

    mAP = evaluation.mAP()
    print('='*60)
    print(f'mAP = {mAP}')


def evaluate():
    cfg, args = parse_test_configs()
    # stereo_args = parse_config()
    cudnn.benchmark = True
    global visualizer

    dataset_root = os.path.join(cfg.dataset_dir, "training")
    KITTI = KittiDataset(dataset_root, mode='val')
    # sfa_model = SFA3D(cfg)

    args_pillars, cfg_pillars = parse_config_pillars()
    pointpillars = PointCloud_3D_Detection(args_pillars, cfg_pillars)

    # KITTI_stereo = KittiDataset(dataset_root, stereo_mode=True)    
    # anynet_model = Stereo_Depth_Estimation(stereo_args,None)

    evaluation = Evaluation(iou_threshold=0.5, evaluate_class=class_name_to_label('Car'), mode=EvalMode.IOU_3D)
    # ======================================================================
    # preds_val = []
    # objects_val = []
    for i in range(args.index, len(KITTI)):
        image, pointcloud, labels, calib = KITTI[i]

        # SFA3D
        # detections = sfa_model.predict(pointcloud)
        # objects = SFA3D_output_to_kitti_objects(detections)

        # # Point Pillars
        pred = pointpillars.predict(pointcloud)
        objects = model_output_to_kitti_objects(pred)

        # preds_val.append(pred)
        # objects_val.append(objects)

        # filter score
        for obj in objects:
            if obj.score < 0.5:
                objects.remove(obj)

        evaluation.evaluate_step(objects, labels, calib)
        if i % 20 == 0:
            print(f'{i}- mAP = {evaluation.mAP()}')

    mAP = evaluation.mAP()
    print('='*60)
    print(f'mAP = {mAP}')

    # with open('objects.pickle', 'wb') as f:
    #     pickle.dump(objects_val, f)
    # with open('predictions.pickle', 'wb') as f:
    #     pickle.dump(preds_val, f)

if __name__ == '__main__':
    evaluate_pickle()

