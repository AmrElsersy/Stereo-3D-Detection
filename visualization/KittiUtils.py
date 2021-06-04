import numpy as np
from enum import Enum
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================
class BBox2D:
    def __init__(self, bbox):
        """
            2D BBox of (x,y) of top left corner
        """
        self.x = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2]
        self.height = bbox[3]
        self.top_left = (self.x,self.y)

class BBox3D:
    def __init__(self, x, y, z, h, w, l, rotation):
        """
            3D BBox with (x,y,z) of center
        """
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

    def volume(self):
        return self.height * self.width * self.length

class Coordinates(Enum):
    CAM_3D_RECT = 0
    CAM_3D_REF = 1
    LIDAR = 2


class VisMode(Enum):
    SCENE_3D = 0
    SCENE_2D = 1

# ================================================
class KittiObject:
    def __init__(self, bbox_3d, label, score=1, bbox_2d=None):
        self.bbox_3d = bbox_3d
        self.label = label
        self.score = score
        self.bbox_2d = bbox_2d

    def __repr__(self):
        return label_to_class_name(self.label) + '(' + str(round(self.score,2)) + ')' + '[' + str(self.label) + '] '

# ================================================
class KittiCalibration:
    """
        Perform different types of calibration between camera & LIDAR

        image = Projection * Camera3D_after_rectification
        image = Projection * R_Rectification * Camera3D_reference

    """
    def __init__(self, calib_path, from_video=False):
        self.calib_path = calib_path
        self.calib_matrix = {}
        if from_video:
            self.calib_matrix = self.parse_calib_from_video(calib_path)
            self.calib_path = os.path.join(calib_path, "modified_calib_file.txt")
            # print('#################', self.calib_path)
        else:
            self.calib_matrix = self.parse_calib_files(calib_path)

        self.P0 = self.calib_matrix["P0"]
        self.P1 = self.calib_matrix["P1"]
        # Projection Matrix (Intrensic) .. from camera 3d (after rectification) to image coord.
        self.P2 = self.calib_matrix["P2"].reshape(3, 4)
        self.P3 = self.calib_matrix["P3"]
        # rectification rotation matrix 3x3
        self.R0_rect = self.calib_matrix["R0_rect"].reshape(3,3)
        # Extrensic Transilation-Rotation Matrix from LIDAR to Cam ref(before rectification)
        self.Tr_velo_to_cam = self.calib_matrix["Tr_velo_to_cam"].reshape(3,4)
        # inverse of Tr_velo_cam
        self.Tr_cam_to_velo = self.inverse_Tr(self.Tr_velo_to_cam)

    def parse_calib_files(self, calib_path):
        assert self.calib_path is not None

        mat_ = {}
        with open(os.path.join(calib_path), 'r') as calib_file:
            for line in calib_file:
                line = line.split()
                # Avoiding empty line exception
                if len(line) == 0:
                    continue
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try: 
                    mat_[line[0][:-1]] = np.array(line[1:], dtype=np.float32)
                except ValueError:
                    continue

        return mat_

    def parse_calib_from_video(self, calib_path):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        assert calib_path is not None

        mat_ = {}
        cam2cam = self.parse_calib_files(
            os.path.join(calib_path, "calib_cam_to_cam.txt")
        )
        velo2cam = self.parse_calib_files(
            os.path.join(calib_path, "calib_velo_to_cam.txt")
        )


        mat_["P0"] = cam2cam["P_rect_00"]
        mat_["P1"] = cam2cam["P_rect_01"]
        mat_["P2"] = cam2cam["P_rect_02"]
        mat_["P3"] = cam2cam["P_rect_03"]

        mat_["R0_rect"] = cam2cam["R_rect_00"]

        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        mat_["Tr_velo_to_cam"] = Tr_velo_to_cam
        return mat_


    def rotx(self, t):
        """ 3D Rotation about the x-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def roty(self, t):
        """ Rotation about the y-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rotz(self, t):
        """ Rotation about the z-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([
            [c,-s, 0], 
            [s, c, 0], 
            [0, 0, 1]])

    def project_lidar_to_image(self, corners=None):
        """
            Projecting a tensor of objects to image plane (u,v)
            P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                        0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                        0,      0,      1,      0]

            image2 coord:
             ----> x-axis (u)
            |
            |
            v y-axis (v)

            velodyne coord:
            front x, left y, up z

            rect/ref camera coord:
            right x, down y, front z

            Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        """
        assert corners is not None

        P_2 = self.P2
        # We're using R_0 as a reference rotation
        # R_0 is expanded to (4,4) by adding a new row & col with zeros & R_0(4,4) = 1
        R_0 = self.R0_rect
        R_0 = np.pad(R_0, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        R_0[-1, -1] = 1

        # Extrinsic transformation matrix from velodyne to camera coordinates
        T_velo_cam = self.Tr_velo_to_cam
        T_velo_cam = np.pad(T_velo_cam, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        T_velo_cam[-1, -1] = 1

        projected_corners = []
        for corner in corners:
            # y_homogeneous = P_2.R_0.T_velo_cam.X_homogeneous
            corner = np.array([corner[0], corner[1], corner[2], 1])
            y = np.dot(T_velo_cam, corner)
            y = np.dot(R_0, y)
            y = np.dot(P_2, y)
            # Updating x, y & z from homogeneous to non homogeneous (u,v)
            z = y[2]
            x = y[0] / z
            y = y[1] / z
            projected_corners.append([x, y, z])

        return np.array(projected_corners, dtype=np.float32)

    def inverse_Tr(self, T):
        """ 
            get inverse of Translation Rotation 4x4 Matrix
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
        'Pedestrian': 0,
        'Person_sitting': 0,

        'Car': 1,
        'Van': 1,
        'Truck': 1,

        'Cyclist': 2,
        
        'Misc' : 0,
        'Tram' : 0
    }
    return class_to_label[classname]

def label_to_class_name(label):
    class_list = ["Pedestrian", "Car", "Cyclist"]
    return class_list[label]

def pillars_labels_to_sfa_labels(label):
    if label == 0:
        return 1
    elif label == 1:
        return 0
    elif label == 2:
        return 2

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
        
        bbox.coordinates = Coordinates.LIDAR
        kitti_object = KittiObject(bbox, label_id)
        kitti_object.score = score
        kitti_objects.append(kitti_object)

    return kitti_objects


def SFA3D_output_to_kitti_objects(detections):
    kitti_objects = []

    for detection in detections:
        cls_id, x, y, z, h, w, l, yaw, score = detection
        
        # z output is shifted down & (l , w) are swaped
        z += h/2
        w, l = l, w

        bbox = BBox3D(x, y, z, h, w, l, yaw)
        bbox.coordinates = Coordinates.LIDAR

        kitti_object = KittiObject(bbox, int(cls_id), score)
        kitti_objects.append(kitti_object)
    
    return kitti_objects

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = torch.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = torch.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = torch.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = torch.tensor([float(x) for x in value.split()], device=device)
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = torch.hstack((pts_3d, torch.ones((n, 1), device=device)))
        return pts_3d_hom

    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return torch.matmul(pts_3d_velo, torch.transpose(self.V2C, 0, 1))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return torch.matmul(pts_3d_ref, torch.transpose(self.C2V, 0, 1))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return torch.transpose(torch.matmul(torch.inverse((self.R0)), torch.transpose(pts_3d_rect, 0, 1)), 0, 1)

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return torch.transpose(torch.matmul(self.R0, torch.transpose(pts_3d_ref, 0, 1)), 0, 1)

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = torch.dot(pts_3d_rect, torch.transpose(self.P, 0, 1))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = torch.zeros((n, 3), device=device)
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = torch.zeros_like(Tr, device=device)  # 3x4
    inv_Tr[0:3, 0:3] = torch.transpose(Tr[0:3, 0:3], 0, 1)
    # print(inv_Tr.shape)
    # print(torch.dot(-inv_Tr[0:3, 0:3], Tr[0:3, 3]).shape)
    inv_Tr[0:3, 3] = torch.matmul(-inv_Tr[0:3, 0:3], Tr[0:3, 3])
    return inv_Tr
