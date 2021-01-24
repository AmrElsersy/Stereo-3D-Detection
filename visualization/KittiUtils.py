import numpy as np
from enum import Enum
# ================================================
class BBox2D:
    def __init__(self, bbox):
        self.x = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2]
        self.height = bbox[3]
        self.top_left = (self.x,self.y)

class BBox3D:
    def __init__(self, x, y, z, h, w, l, rotation):
        self.pos = (x,y,z)
        self.dims = (h,w,l)
        self.x = x
        self.y = y
        self.z = z
        self.height = h # z length 20
        self.width  = w # y length 10 
        self.length = l # x length 50
        self.rotation = rotation
        # default coordinates .. same as model output
        self.coordinates = Coordinates.LIDAR

class Coordinates(Enum):
    CAM_3D_RECT = 0
    CAM_3D_REF = 1
    LIDAR = 2

# ================================================
class KittiObject:
    def __init__(self, bbox_3d, label, score=1, bbox_2d=None):
        self.bbox_3d = bbox_3d
        self.label = label
        self.score = score
        self.bbox_2d = bbox_2d


# ================================================
class KittiCalibration:
    """
        Perform different types of calibration between camera & LIDAR

        image = Projection * Camera3D_after_rectification
        image = Projection * R_Rectification * Camera3D_reference

    """
    def __init__(self, calib_dict):
        self.P0 = np.asarray(calib_dict["P0"])
        self.P1 = np.asarray(calib_dict["P1"])
        self.P3 = np.asarray(calib_dict["P3"])
        # Projection Matrix (Intrensic) .. from camera 3d (after rectification) to image coord.
        self.P2 = np.asarray(calib_dict["P2"]).reshape(3, 4)
        # rectification rotation matrix 3x3
        self.R0_rect = np.asarray(calib_dict["R0_rect"]).reshape(3,3)
        # Extrensic Transilation-Rotation Matrix from LIDAR to Cam ref(before rectification) 
        self.Tr_velo_to_cam = np.asarray(calib_dict["Tr_velo_to_cam"]).reshape(3,4)
        # inverse of Tr_velo_cam
        self.Tr_cam_to_velo = self.inverse_Tr(self.Tr_velo_to_cam)

    def inverse_Tr(self, T):
        """ 
            get inverse of Transilation Rotation 4x4 Matrix
            Args:
                T: 4x4 Matrix
                    ([
                        [R(3x3) t],
                        [0 0 0  1]
                    ])
            Return:
                Inverse: 4x4 Matrix
                    ([
                        [R^-1   -R^1 * t],
                        [0 0 0         1]
                    ])                
        """
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        # print(R.shape, t.shape)
        R_inv = np.linalg.inv(R)
        t_inv = np.dot(-R_inv, t).reshape(3,1)
        T_inv = np.hstack((R_inv, t_inv))
        T_inv = np.vstack( (T_inv, np.array([0,0,0,1])) )
        return T_inv

    def rectified_camera_to_velodyne(self, points):
        """
            Converts 3D Box in Camera coordinates(after rectification) to 3D Velodyne coordinates
            Args: points
                numpy array (N, 3) in cam coord, N is points number 
            return: 
                numpy array (N, 3) in velo coord.
        """

        # from rectified cam to ref cam 3d
        R_rect_inv = np.linalg.inv(self.R0_rect)
        points_ref =  np.dot(R_rect_inv, points.T) # 3xN

        # add homogenious 4xN
        points_ref = np.vstack((points_ref, np.ones((1, points_ref.shape[1]), dtype=np.float)))

        # velodyne = ref_to_velo * points_ref
        points_3d_velodyne = np.dot(self.Tr_cam_to_velo, points_ref)

        return points_3d_velodyne.T






# ================================================
def class_name_to_label(classname):
    class_to_label = {
        'Car': 0,
        'Van': 0,
        'Truck': 0,
        'Pedestrian': 1,
        'Person_sitting': 1,
        'Cyclist': 2,
        'Misc' : 0
    }
    return class_to_label[classname]

def label_to_class_name(label):
    class_list = ["Car", "Pedestrian", "Cyclist"]
    return class_list[label]

def model_output_to_kitti_objects(pred_dict):
    kitti_objects = []
    
    # tensor
    boxes  = pred_dict[0]["pred_boxes"]
    scores = pred_dict[0]["pred_scores"]
    labels = pred_dict[0]["pred_labels"]

    # convert cuda tensor to numpy array / list
    boxes  = boxes.cpu().numpy()
    scores = scores.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()

    n_objects = boxes.shape[0]

    for i in range(n_objects):
        # bbox .. x,y,z, dx,dy,dz, rotation
        x, y, z, w, l, h, rot = boxes[i].tolist()
        bbox = BBox3D(x, y, z, h, w, l, rot)
        # score
        score = scores[i]
        # label index is shifted in model output
        label_id = labels[i] - 1

        kitti_object = KittiObject(bbox, label_id)
        kitti_objects.append(kitti_object)

    return kitti_objects
