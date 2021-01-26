import numpy as np
from math import sin, cos, radians
from KittiDataset import KittiDataset
from KittiUtils import *
# import BEVutils as BEVutils
import cv2

from torch import tensor
# from mayavi import mlab

class KittiVisualizer:
    def __init__(self):
        # self.figure = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1), size=(1280, 720))
        pass

    def show(self):
        mlab.show(stop=True)

    def visualize(self, pointcloud, objects, labels=None, calib=None):
        """
            Visualize the Scene including Point Cloud & 3D Boxes 

            Args:
                pointcloud: numpy array (points_n, 3)
                objects: list of KittiObject represents model output
                labels: list of KittiObjects represents dataset labels
                calib: Kitti Calibration Object (must be specified if you pass boxes with cam_rect_coord)
        """
        # Point Cloud
        self.visuallize_pointcloud(pointcloud)

        # 3D Boxes of model output
        for obj in objects:
            bbox_3d = obj.bbox_3d
            color = self.__get_box_color(obj.label)
            self.visualize_3d_bbox(bbox=bbox_3d, color=color, calib=calib)

            self.__draw_text(*bbox_3d.pos, text=str(obj.score), color=color)

        # 3D Boxes of dataset labels 
        if labels is not None:
            for obj in labels:
                self.visualize_3d_bbox(obj.bbox_3d, (1,0,0), calib)

    def visualize_bev(self, pointcloud, objects, labels=None, calib=None):
        BEV = BEVutils.pointcloud_to_bev(pointcloud)
        BEV = self.__bev_to_colored_bev(BEV)

        # cv2.imshow("BEV0", BEV[:,:,0])
        # cv2.imshow("BEV1", BEV[:,:,1])
        # cv2.imshow("BEV2", BEV[:,:,2])

        # clip boxes
        objects = BEVutils.clip_3d_boxes(objects, calib)
        
        # 3D Boxes of model output
        for obj in objects:
            color = self.__get_box_color(obj.label)
            self.__draw_bev_box3d(BEV, obj.bbox_3d, color, calib)

        # # 3D Boxes of dataset labels 
        if labels is not None:
            labels = BEVutils.clip_3d_boxes(labels, calib)
            for obj in labels:
                self.__draw_bev_box3d(BEV, obj.bbox_3d, (1,0,0), calib)

        cv2.imshow("BEV", BEV)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        mlab.close(all=True)
    
    def __draw_bev_box3d(self, bev, bbox_3d, color, calib):
        corners = self.__convert_3d_bbox_to_corners(bbox_3d, calib)
        c0 = BEVutils.corner_to_bev_coord(corners[0])
        c1 = BEVutils.corner_to_bev_coord(corners[1])
        c2 = BEVutils.corner_to_bev_coord(corners[2])
        c3 = BEVutils.corner_to_bev_coord(corners[3])

        cv2.line(bev, (c0[0], c0[1]), (c1[0], c1[1]), color, 1)
        cv2.line(bev, (c0[0], c0[1]), (c2[0], c2[1]), color, 1)
        cv2.line(bev, (c3[0], c3[1]), (c1[0], c1[1]), color, 1)
        cv2.line(bev, (c3[0], c3[1]), (c2[0], c2[1]), color, 1)

    def __bev_to_colored_bev(self, bev):
        intensity_map = bev[:,:,0] * 255
        height_map = bev[:,:,1]
        np.set_printoptions(threshold=np.inf)

        minZ = BEVutils.boundary["minZ"]
        maxZ = BEVutils.boundary["maxZ"]
        height_map = 255 - 255 * (height_map - minZ) / (maxZ - minZ) 

        # make empty points black in all channels
        empty_points_indices = np.where(intensity_map == 0)
        height_map[empty_points_indices] = 0

        BEV = np.dstack((height_map, height_map, intensity_map))
        return BEV

    def visuallize_pointcloud(self, pointcloud):
        pointcloud = self.__to_numpy(pointcloud)
        mlab.points3d(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], 
                    colormap='gnuplot', scale_factor=1, mode="point",  figure=self.figure)
        self.__draw_axes()

    def __draw_axes(self):
        l = 4 # axis_length
        w = 1
        mlab.plot3d([0, l], [0, 0], [0, 0], color=(0, 0, 1), line_width=w, figure=self.figure) # x
        mlab.plot3d([0, 0], [0, l], [0, 0], color=(0, 1, 0), line_width=w, figure=self.figure) # y
        mlab.plot3d([0, 0], [0, 0], [0, l], color=(1, 0, 0), line_width=w, figure=self.figure) # z

    def visualize_3d_bbox(self, bbox: BBox3D, color=(0,1,0), calib=None):
        corners = self.__convert_3d_bbox_to_corners(bbox, calib)
        self.__draw_box_corners(corners, color)

    def visualize_2D_image(self, image, kitti_objects, calib):
        self.current_image = image

        for object in kitti_objects:
            if object.score < 0.7:
                continue
            corners = self.__convert_3d_bbox_to_corners(object.bbox_3d, calib)
            proj_corners = calib.project_lidar_to_image(corners)
            color = self.__get_box_color(object.label)
            self.__draw_box_corners(proj_corners, color, VisMode.SCENE_2D)

            # Draw score
            # point = proj_corners[2].astype(np.int32)
            # score_point = (point[0], point[1]-10)
            # score_per_box = int(object.score * 100)
            # self.__draw_text_2D(f"Score: {score_per_box}", point)

            # Draw label
            # label_point = (point[0], point[1]-20)
            # self.__draw_text_2D(f"{object.label}", point)

        cv2.imshow('WINDOW',self.current_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def __convert_3d_bbox_to_corners(self, bbox: BBox3D, calib=None):
        """
            convert BBox3D with x,y,z, width, height, depth .. to 8 corners
                    h
              3 -------- 1
          w  /|         /|
            2 -------- 0 . d
            | |        | |
            . 7 -------- 5
            |/         |/
            6 -------- 4

                        z    x
                        |   / 
                        |  /
                        | /
                y--------/
        """
        x = bbox.x
        y = bbox.y
        z = bbox.z
        w = bbox.width  # y
        h = bbox.height # z
        l = bbox.length # x
        angle = bbox.rotation

        # convert from Camera 3D coordinates to LIDAR coordinates.
        if bbox.coordinates != Coordinates.LIDAR:
            if calib is None:
                print("WARNING: Visualization is in LIDAR coord & you pass a bbox of camera coord")

            point = np.array([x, y, z]).reshape(1,3)

            # convert x, y, z from rectified cam coordinates to velodyne coordinates
            if bbox.coordinates == Coordinates.CAM_3D_RECT:
                point = calib.rectified_camera_to_velodyne(point)

            x = point[0,0]
            y = point[0,1] 
            # model output is z center but dataset annotations consider z at the bottom 
            z = point[0,2] + h/2
        
        # convert (x,y,z) from center to top left corner (corner 0)
        x = x - w/2
        y = y - l/2
        z = z + h/2

        top_corners = np.array([
            [x, y, z],
            [x+w, y, z],
            [x, y+l, z],
            [x+w, y+l, z]
        ])

        # same coordinates but z = z_top - box_height
        bottom_corners = top_corners - np.array([0,0, h])

        # concatinate 
        corners = np.concatenate((top_corners,bottom_corners), axis=0)

        # 3x3 Rotation Matrix along z 
        R = calib.rotz(angle)

        # Translate the box to origin to perform rotation
        center = np.array([x+w/2, y+l/2, z-h/2])
        centered_corners = corners - center

        # Rotate
        rotated_corners = np.dot( R, centered_corners.T ).T

        # Translate it back to its position
        corners = rotated_corners + center

        # output of sin & cos sometimes is e-17 instead of 0
        corners = np.round(corners, decimals=10)

        return corners

    def __draw_box_corners(self, corners, clr, vis_mode=VisMode.SCENE_3D):
        if corners.shape[0] != 8:
            print("Invalid box format")
            return

        c0 = corners[0]
        c1 = corners[1] 
        c2 = corners[2] 
        c3 = corners[3] 
        c4 = corners[4] 
        c5 = corners[5] 
        c6 = corners[6] 
        c7 = corners[7] 

        # top suqare
        self.__draw_line(c0, c1, clr, vis_mode)
        self.__draw_line(c0, c2, clr, vis_mode)
        self.__draw_line(c3, c1, clr, vis_mode)
        self.__draw_line(c3, c2, clr, vis_mode)
        # bottom square
        self.__draw_line(c4, c5, clr, vis_mode)
        self.__draw_line(c4, c6, clr, vis_mode)
        self.__draw_line(c7, c5, clr, vis_mode)
        self.__draw_line(c7, c6, clr, vis_mode)
        # vertical edges
        self.__draw_line(c0, c4, clr, vis_mode)
        self.__draw_line(c1, c5, clr, vis_mode)
        self.__draw_line(c2, c6, clr, vis_mode)
        self.__draw_line(c3, c7, clr, vis_mode)

    def __draw_line(self, corner1, corner2, clr,vis_mode):
        x = 0
        y = 1
        z = 2
        if vis_mode == VisMode.SCENE_3D:
            mlab.plot3d([corner1[x], corner2[x]], [corner1[y], corner2[y]], [corner1[z], corner2[z]],
                    line_width=2, color=clr, figure=self.figure)

        elif vis_mode == VisMode.SCENE_2D:

            cv2.line(self.current_image, (corner1[x], corner1[y]), (corner2[x], corner2[y]), color=tuple([255 * x for x in clr]), thickness=2)

        else:
            pass

    def __draw_text_2D(self, text, point, color=(0, 255, 0), font_scale=0.6, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        cv2.putText(self.current_image, text, point, font, font_scale, color, thickness)

    def __draw_text(self, x, y, z, text, color):
        mlab.text3d(0,0,0, text, scale=0.3, color=color, figure=self.figure)
    
    def __get_box_color(self, class_id):
        if type(class_id) == str:
            class_id = class_name_to_label(class_id)

        colors = [
            (0,1,0),
            (0,0,1),
            (0,1,1),
        ]

        return colors[class_id]

    def __to_numpy(self, pointcloud):
        if not isinstance(pointcloud, np.ndarray):
            return pointcloud.cpu().numpy()
        return pointcloud


KITTI = KittiDataset('/home/ayman/FOE-Linux/Graduation_Project/KITTI/training')
image, pointcloud, objects, calib = KITTI[10]
visualizer = KittiVisualizer()
# visualizer.visualize_2D_image(image, objects, calib)


pred = [{'pred_boxes': tensor([[ 5.5041e+00, -4.3590e+00, -1.0023e+00,  3.3916e+00,  1.6107e+00,
          1.4861e+00,  6.1331e+00],
        [ 1.2106e+01,  2.4439e+00, -8.8113e-01,  3.9857e+00,  1.6854e+00,
          1.4671e+00,  3.0013e+00],
        [ 1.6783e+01, -5.8833e+00, -8.6557e-01,  3.1027e+00,  1.4880e+00,
          1.4869e+00,  6.1249e+00],
        [ 2.3967e+01,  3.6035e-01, -7.6771e-01,  3.9717e+00,  1.6540e+00,
          1.5694e+00,  2.9736e+00],
        [ 2.1767e+01,  2.6778e+01, -7.4806e-01,  3.2355e+00,  1.4420e+00,
          1.4612e+00,  6.1272e+00],
        [ 4.3203e+01, -4.4596e+00, -6.5152e-01,  3.2930e+00,  1.5471e+00,
          1.5108e+00,  2.8025e+00],
        [ 2.2139e+01, -6.7291e+00, -8.7812e-01,  3.6720e+00,  1.6025e+00,
          1.4315e+00,  6.1017e+00],
        [ 2.9364e+01, -6.5818e-01, -7.9045e-01,  3.5215e+00,  1.5284e+00,
          1.5251e+00,  2.9503e+00],
        [ 2.0152e+00,  5.6060e+00, -9.4099e-01,  4.5025e+00,  1.7946e+00,
          1.6852e+00,  2.2784e+00],
        [ 2.8571e+01, -7.8730e+00, -8.2800e-01,  3.8899e+00,  1.6292e+00,
          1.4490e+00,  6.1300e+00],
        [ 2.3525e+01, -8.4265e+00, -5.7474e-01,  8.3250e-01,  6.5383e-01,
          1.8969e+00,  6.3983e+00],
        [ 5.2723e+01, -1.1171e+01, -4.9946e-01,  3.9322e+00,  1.6034e+00,
          1.6057e+00,  2.5870e+00],
        [ 3.6645e+01, -1.8423e+00, -7.8137e-01,  3.9124e+00,  1.5857e+00,
          1.5188e+00,  2.9589e+00],
        [ 1.4589e+01, -8.1472e+00, -6.8846e-01,  8.6483e-01,  7.3425e-01,
          1.8432e+00,  2.8800e+00],
        [ 3.6217e+01, -9.6881e+00, -7.3354e-01,  3.5871e+00,  1.6406e+00,
          1.5077e+00,  6.0456e+00],
        [ 6.7796e+01, -2.6863e+01, -6.1149e-02,  3.8945e+00,  1.6270e+00,
          1.4926e+00,  5.7557e+00],
        [ 3.0238e+00,  1.2788e+01, -8.8431e-01,  3.9051e+00,  1.5595e+00,
          1.4319e+00,  3.6127e+00],
        [ 2.2918e+01, -1.9455e+01, -7.0166e-01,  5.2463e-01,  6.3954e-01,
          1.6918e+00,  5.7155e+00],
        [ 4.4231e+00, -1.7600e+01, -5.4812e-01,  6.9418e-01,  5.9308e-01,
          1.7122e+00,  5.6545e+00],
        [ 1.2553e+01, -3.4548e+01,  3.1960e-01,  9.5871e-01,  6.8724e-01,
          1.8605e+00,  2.4200e+00],
        [ 2.8862e+00, -1.1236e+01, -6.3180e-01,  1.7843e+00,  5.6750e-01,
          1.8307e+00,  4.4822e+00],
        [ 1.4180e+01,  1.0826e+01, -4.7658e-01,  8.9363e-01,  5.3882e-01,
          1.9344e+00,  5.4969e+00],
        [ 1.3046e+01, -2.4551e+01,  4.6369e-03,  1.6417e+00,  5.6287e-01,
          1.6026e+00,  5.8465e+00],
        [ 1.8490e+01, -2.7844e+01, -2.2200e-01,  3.5863e+00,  1.5388e+00,
          1.3792e+00,  6.6197e+00],
        [ 3.1549e+01,  1.4402e+01, -6.7287e-01,  3.9653e-01,  6.5481e-01,
          1.7926e+00,  6.7428e+00],
        [ 6.1442e+00,  9.0741e+00, -4.7173e-01,  1.1989e+00,  6.9589e-01,
          2.0958e+00,  3.6548e+00],
        [ 1.6762e+01, -4.1032e+01,  4.3824e-01,  4.0478e+00,  1.6105e+00,
          1.5979e+00,  5.6952e+00],
        [ 2.0605e+01, -2.9757e+01,  1.7164e-01,  2.3616e-01,  4.7174e-01,
          1.7787e+00,  1.5147e+00],
        [ 2.3492e+01, -2.6307e+01, -3.9680e-01,  4.2710e+00,  1.6441e+00,
          1.4501e+00,  1.1704e+00],
        [ 6.3628e+01, -2.4218e-01, -2.1012e-01,  8.9807e-01,  5.0813e-01,
          1.9069e+00,  4.2939e+00],
        [ 2.1802e+01,  9.6388e+00, -6.0165e-01,  1.8918e+00,  6.9331e-01,
          1.7963e+00,  4.8379e+00],
        [ 4.6014e+01, -1.3344e+01, -5.6706e-01,  3.6812e+00,  1.6137e+00,
          1.4589e+00,  2.5625e+00],
        [ 8.3010e+00,  3.4845e+01, -8.9965e-01,  3.6919e+00,  1.5254e+00,
          1.4489e+00,  6.4016e+00],
        [ 5.4816e+01,  4.0298e+01, -4.8520e-01,  3.9901e+00,  1.6678e+00,
          1.7106e+00,  1.2233e+00],
        [ 5.2763e-01, -5.8049e-01, -1.0801e+00,  4.0868e+00,  1.6635e+00,
          1.5774e+00,  5.9765e+00],
        [ 1.4723e+01, -3.3088e+01,  4.0932e-01,  4.1721e+00,  1.6003e+00,
          1.5213e+00,  2.1954e+00],
        [ 4.9989e+01, -1.6209e+01, -2.2394e-01,  4.0972e+00,  1.6198e+00,
          1.6493e+00,  5.7883e+00],
        [ 5.2673e+01,  3.3614e+00, -6.3586e-01,  4.1216e+00,  1.6379e+00,
          1.5128e+00,  3.0718e+00],
        [ 2.5096e+01, -1.6287e+01, -8.3649e-01,  5.0648e-01,  6.6092e-01,
          1.6190e+00,  6.2754e+00],
        [ 1.7638e+01, -2.5884e+01, -2.7044e-02,  5.7649e-01,  6.2769e-01,
          1.7381e+00,  2.1762e+00],
        [ 1.5118e+01, -2.5276e+01, -1.9550e-02,  3.5740e+00,  1.5984e+00,
          1.4769e+00,  3.4702e+00],
        [ 1.1851e+01, -3.0782e+01, -9.3451e-03,  3.6586e+00,  1.5567e+00,
          1.3937e+00,  4.8127e+00],
        [ 4.7865e+01,  4.0911e+01, -1.0034e+00,  3.9714e+00,  1.6381e+00,
          1.5135e+00,  3.8077e+00],
        [ 2.2183e+01, -3.8017e+01,  6.5019e-01,  5.5555e-01,  5.8220e-01,
          1.7876e+00,  2.1689e+00],
        [ 1.1330e+01, -2.2054e+01, -2.7680e-01,  6.1404e-01,  7.1226e-01,
          1.7740e+00,  5.6314e+00],
        [ 2.7068e+01, -1.3022e+01, -8.0845e-01,  5.4416e-01,  6.0143e-01,
          1.6511e+00,  6.0598e+00],
        [ 7.6096e+00, -3.1960e+01, -1.3805e-02,  3.7690e+00,  1.6778e+00,
          1.5192e+00,  2.5607e+00],
        [ 2.1468e+01, -1.8873e+01, -4.6003e-01,  5.2208e-01,  5.3762e-01,
          1.6662e+00,  2.7601e+00]], device='cuda:0'), 'pred_scores': tensor([0.9750, 0.9672, 0.9421, 0.9217, 0.9027, 0.8336, 0.8110, 0.7551, 0.6883,
        0.6364, 0.5867, 0.5532, 0.5186, 0.4033, 0.3907, 0.3135, 0.2839, 0.2810,
        0.2741, 0.2640, 0.2569, 0.2410, 0.2286, 0.2225, 0.2115, 0.1963, 0.1858,
        0.1791, 0.1627, 0.1543, 0.1470, 0.1447, 0.1431, 0.1417, 0.1386, 0.1367,
        0.1261, 0.1235, 0.1231, 0.1225, 0.1217, 0.1190, 0.1155, 0.1121, 0.1059,
        0.1050, 0.1040, 0.1019], device='cuda:0'), 'pred_labels': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 2, 3, 1,
        2, 2, 1, 2, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2],
       device='cuda:0')}]


objects = model_output_to_kitti_objects(pred)
visualizer.visualize_2D_image(image, objects, calib)
# visualizer.visualize(pointcloud, objects)
# visualizer.visualize(pointcloud, objects, labels, calib)
# visualizer.visualize_bev(pointcloud, objects, labels, calib=calib)

# visualizer.show()


 
# # import open3d
# class Open3dVisualizer:
#     def visuallize_pcd(self, path):
#         pcd = open3d.io.read_point_cloud(path)
#         pcd_points = np.asarray(pcd.points)
#         print(pcd)
#         return pcd_points
#     def visuallize_pointcloud(self, pointcloud):
#         # convert numpy pointclod to open3d pointcloud
#         points = open3d.utility.Vector3dVector()
#         points.extend(pointcloud)
#         open3d_pointcloud = open3d.geometry.PointCloud(points)
#         print(open3d_pointcloud)
#         # visuallize
#         open3d.visualization.draw_geometries([open3d_pointcloud])
# ==================== Open3D ====================
# visualizer = Open3dVisualizer()
# p = visualizer.visuallize_pcd("/home/amrelsersy/KITTI/pcd/0000000000.pcd")
# visualizer.visuallize_pointcloud(p)
# PCD
# pcd = open3d.io.read_point_cloud("/home/amrelsersy/KITTI/pcd/0000000000.pcd")
# pcd_points = np.asarray(pcd.points)
# print(pcd_points.shape)







