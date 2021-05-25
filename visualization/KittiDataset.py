import PIL
from PIL import Image
import os
from torch.utils.data import Dataset 
import numpy as np
import cv2

from visualization.KittiUtils import *
from PIL import Image

class KittiDataset(Dataset):
    def __init__(self, root="../data/kitti/testing", mode='train', transform = None, stereo_mode=False):
        self.root = root
        self.stereo_mode = stereo_mode
        self.mode = mode

        start_idx = 0
        end_idx = -1
        if mode == 'train':
            end_idx = 6000
        elif mode == 'val':
            start_idx = 6000
        else:
            raise ValueError()

        self.rootPointclouds = os.path.join(self.root, "velodyne")
        self.rootImages = os.path.join(self.root, "image_2")
        self.rootAnnotations = os.path.join(self.root, "label_2")
        self.rootCalibration = os.path.join(self.root, "calib")

        self.imagesNames      = sorted(os.listdir(self.rootImages)) [start_idx : end_idx]
        self.pointCloudNames  = sorted(os.listdir(self.rootPointclouds))[start_idx : end_idx]
        self.annotationNames  = sorted(os.listdir(self.rootAnnotations))[start_idx : end_idx]
        self.calibrationNames = sorted(os.listdir(self.rootCalibration))[start_idx : end_idx]

        if self.stereo_mode:
            self.rootRightImages = os.path.join(self.root, "image_3")
            self.rightImagesNames = sorted(os.listdir(self.rootRightImages))[start_idx : end_idx]
        
    def __getitem__(self, index):
        imagePath = os.path.join(self.rootImages, self.imagesNames[index])
        pointcloudPath = os.path.join(self.rootPointclouds, self.pointCloudNames[index])
        annotationPath = os.path.join(self.rootAnnotations, self.annotationNames[index])
        calibrationPath = os.path.join(self.rootCalibration, self.calibrationNames[index])

        image = self.read_image_cv2(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pointcloud = self.read_pointcloud_bin(pointcloudPath)
        labels = self.read_labels_annotations(annotationPath)
        labels = self.convert_to_kitti_objects(labels)
        calib = KittiCalibration(calib_path=calibrationPath)

        if self.stereo_mode:
            rightImagesPath = os.path.join(self.rootRightImages, self.rightImagesNames[index])
            rightImage = self.read_image_cv2(rightImagesPath)

            return image, rightImage, labels, calib

        return image, pointcloud, labels, calib

    def __len__(self):
        return len(self.annotationNames)

    def read_pointcloud_bin(self, path):
        # read .bin and convert to tensor
        pointCloud = np.fromfile(path, dtype=np.float32)
        # reshape to get each point
        pointCloud = pointCloud.reshape(-1, 4)

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


class KittiVideo:
    """ Load data for KITTI videos """

    def __init__(self, imgL_dir, imgR_dir, lidar_dir, calib_dir):
        self.calib = KittiCalibration(calib_path=calib_dir, from_video=True)
        self.imgL_dir = imgL_dir
        self.imgR_dir = imgR_dir
        self.lidar_dir = lidar_dir

        self.imgL_filenames = sorted(
            [os.path.join(imgL_dir, filename) for filename in os.listdir(imgL_dir)]
        )
        
        self.imgR_filenames = sorted(
            [os.path.join(imgR_dir, filename) for filename in os.listdir(imgR_dir)]
        )

        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )

        assert(len(self.imgL_filenames) == len(self.imgR_filenames))
        assert(len(self.imgL_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.imgL_filenames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        imgL, imgR = self.__get_image(index) 
        return imgL, imgR, self.__get_lidar(index), self.__get_calibration()

    def __get_image(self, index):
        assert index < self.num_samples
        imgL_filename = self.imgL_filenames[index]
        imgR_filename = self.imgR_filenames[index]
        return cv2.imread(imgL_filename), cv2.imread(imgR_filename)
    
    def __get_lidar(self, index):
        assert index < self.num_samples
        lidar_filename = self.lidar_filenames[index]

        scan = np.fromfile(lidar_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def __get_calibration(self):
        return self.calib
