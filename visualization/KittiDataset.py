import PIL
from PIL import Image
import os
from torch.utils.data import Dataset 
import numpy as np
import cv2

from visualization.KittiUtils import *

class KittiDataset(Dataset):
    def __init__(self, root="/home/amrelsersy/KITTI", transform = None):
        self.root = root
        
        self.rootPointclouds = os.path.join(self.root, "velodyne")
        self.rootImages = os.path.join(self.root, "image_2")
        self.rootAnnotations = os.path.join(self.root, "label_2")
        self.rootCalibration = os.path.join(self.root, "calib")

        self.imagesNames = sorted(os.listdir(self.rootImages))
        self.pointCloudNames = sorted(os.listdir(self.rootPointclouds))
        self.annotationNames = sorted(os.listdir(self.rootAnnotations))
        self.calibrationNames = sorted(os.listdir(self.rootCalibration))

        # - Camera:   x: right,   y: down,  z: forward
        # - Velodyne: x: forward, y: left,  z: up
        
    def __getitem__(self, index):
        imagePath = os.path.join(self.rootImages, self.imagesNames[index])
        pointcloudPath = os.path.join(self.rootPointclouds, self.pointCloudNames[index])
        annotationPath = os.path.join(self.rootAnnotations, self.annotationNames[index])
        calibrationPath = os.path.join(self.rootCalibration, self.calibrationNames[index])


        image = self.read_image_cv2(imagePath)
        pointcloud = self.read_pointcloud_bin(pointcloudPath)
        labels = self.read_labels_annotations(annotationPath)
        calibrations = self.read_calibrations(calibrationPath)

        calibrations = self.convert_to_kitti_calib(calibrations)
        labels = self.convert_to_kitti_objects(labels)

        return image, pointcloud, labels, calibrations

    def __len__(self):
        return len(self.pointCloudNames)

    def read_pointcloud_bin(self, path):
        # read .bin and convert to tensor
        pointCloud = np.fromfile(path, dtype=np.float32)
        # reshape to get each point
        pointCloud = pointCloud.reshape(-1, 4)
        # we don't need reflectivity (4th dim in point)
        pointCloud = pointCloud[:,:3]

        return pointCloud

    def read_image_pil(self, path):
        image = Image.open(path)
        # print(image.format, image.mode, image.size)
        return image

    def read_image_cv2(self, path):
        pil_image = self.read_image_pil(path)
        return self.convert_pil_cv2(pil_image)
    
    def convert_pil_cv2(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def read_labels_annotations(self, path):
        annotationFile = open(path, "r")
        annotationLines = annotationFile.read().splitlines()
        # labels of frame
        labels = []

        for line in annotationLines:
            annotations = line.split(" ")

            class_name = annotations[0]

            if class_name == "DontCare":
                continue

            # 0 (non-truncasted) to 1 (truncated), where truncated refers to the object leaving image boundaries
            leaving_img_boundry = annotations[1]
            # indicating occlusion state:0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
            occluded = annotations[2] 
            # range [-pi, pi]
            observation_angle = annotations[3] 
            # left - top - right - bottom .. image coordinates
            bbox_2d = annotations[4:8] 
            # 3D Dims height - width - length ... 3D Camera Coordinates (height(z), width(y), length(x))
            bbox_3d_dims = annotations[8:11] 
            # 3D Position in 3D Camera rectified Coordinates (x,y,z) (center of object)
            bbox_3d_pos = annotations[11:14] 
            # Rotation ry around Y-axis in camera coordinates [-pi..pi]
            rotation_y = annotations[14] 
            # class_score = annotations[15] # class score

            bbox_2d = [float(dim) for dim in bbox_2d]
            bbox_3d_dims = [float(dim) for dim in bbox_3d_dims]
            bbox_3d_pos = [float(dim) for dim in bbox_3d_pos]
            
            width = bbox_2d[2] - bbox_2d[0]
            height = bbox_2d[3] - bbox_2d[1]

            # x,y (of top left) , width , height
            bbox_2d = [bbox_2d[0] , bbox_2d[1], width, height] 
            bbox_3d = [bbox_3d_pos, bbox_3d_dims, float(rotation_y)] 

            # print(bbox_2d)
            # print(bbox_3d)

            label = {}
            label["bbox_2d"] = bbox_2d
            label["bbox_3d"] = bbox_3d
            label["class_name"] = class_name
            # label["score"] = class_score

            labels.append(label)

        return labels

    def read_calibrations(self, path):
        calibrationFile = open(path, "r")
        calibrationLines = calibrationFile.read().splitlines()

        calibrations = [line.split(" ") for line in calibrationLines]

        calib_dict = {}

        for calib in calibrations:
            # remove the ":" at last
            calib_name = calib[0][:-1] 
            # convert the rest of the line to float ... then convert the list to np array
            calib_dict[calib_name] = np.array( [ float(c) for c in calib[1:] ] )

        return calib_dict

    def convert_to_kitti_calib(self, calib_dict):
        return KittiCalibration(calib_dict)

    def convert_to_kitti_objects(self, labels):
        """
            args:
                labels: list of labels dictionaries
            return:
                list of Kitti Objects
        """

        kitti_labels = []
        for label in labels:
            bbox_2d = label["bbox_2d"]
            bbox_3d = label["bbox_3d"]
            label_id = class_name_to_label(label["class_name"])

            pos_3d = bbox_3d[0]
            dims_3d = bbox_3d[1]
            rotation = bbox_3d[2]

            bbox_3d = BBox3D(*pos_3d, *dims_3d, rotation)        
            bbox_2d = BBox2D(bbox_2d)

            # coordinates in dataset is camera rect 3D unlike the model output (lidar coord)
            bbox_3d.coordinates = Coordinates.CAM_3D_RECT

            kitti_object =  KittiObject(bbox_3d=bbox_3d, label=label_id, bbox_2d=bbox_2d) 
            kitti_labels.append(kitti_object)

        return kitti_labels


# KITTI = KittiDataset()
# _, pointcloud, label, calib = KITTI[1]
# print(pointcloud.shape)
# print(label)
