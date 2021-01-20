import numpy as np
from math import sin, cos, radians
from visualization.KittiDataset import KittiDataset
from visualization.KittiUtils import * 

from torch import tensor
from mayavi import mlab

class KittiVisualizer:
    def __init__(self):
        self.figure = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1), size=(1280, 720))

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
            color = self.get_box_color(obj.label)
            self.visualize_3d_bbox(bbox=bbox_3d, color=color, calib=calib)

            self.draw_text(*bbox_3d.pos, text=str(obj.score), color=color)

        # 3D Boxes of dataset labels 
        if labels is not None:
            for obj in labels:
                self.visualize_3d_bbox(obj.bbox_3d, (1,0,0), calib)


    def visuallize_pointcloud(self, pointcloud):
        pointcloud = self.to_numpy(pointcloud)
        mlab.points3d(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], 
                    colormap='gnuplot', scale_factor=1, mode="point",  figure=self.figure)
        self.draw_axes()

    def draw_axes(self):
        l = 4 # axis_length
        w = 1
        mlab.plot3d([0, l], [0, 0], [0, 0], color=(0, 0, 1), line_width=w, figure=self.figure) # x
        mlab.plot3d([0, 0], [0, l], [0, 0], color=(0, 1, 0), line_width=w, figure=self.figure) # y
        mlab.plot3d([0, 0], [0, 0], [0, l], color=(1, 0, 0), line_width=w, figure=self.figure) # z

    def visualize_3d_bbox(self, bbox: BBox3D, color=(0,1,0), calib=None):
        corners = self.convert_3d_bbox_to_corners(bbox, calib)
        self.draw_box_corners(corners, color)

    def convert_3d_bbox_to_corners(self, bbox: BBox3D, calib=None):
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
            elif bbox.coordinates == Coordinates.CAM_3D_REF:
                point = calib.camera_ref_to_velodyne(point)        
            
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

        # ======== Rotation ========          
        cosa = cos(angle)
        sina = sin(angle)

        # 3x3 Rotation Matrix along z 
        R = np.array([
            [ cosa, sina, 0],
            [-sina, cosa, 0],
            [ 0,    0,    1]
        ])

        # Translate the box to origin to perform rotation
        center = np.array([x+w/2, y+l/2, 0])
        centered_corners = corners - center

        # Rotate
        rotated_corners = np.dot( R, centered_corners.T ).T

        # Translate it back to its position
        corners = rotated_corners + center

        # output of sin & cos sometimes is e-17 instead of 0
        corners = np.round(corners, decimals=10)

        return corners

    def draw_box_corners(self, corners, clr):
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
        self.draw_line(c0, c1, clr)
        self.draw_line(c0, c2, clr)
        self.draw_line(c3, c1, clr)
        self.draw_line(c3, c2, clr)
        # bottom square
        self.draw_line(c4, c5, clr)
        self.draw_line(c4, c6, clr)
        self.draw_line(c7, c5, clr)
        self.draw_line(c7, c6, clr)
        # vertical edges
        self.draw_line(c0, c4, clr)
        self.draw_line(c1, c5, clr)
        self.draw_line(c2, c6, clr)
        self.draw_line(c3, c7, clr)

    def draw_line(self, corner1, corner2, clr):
        x = 0
        y = 1
        z = 2
        mlab.plot3d([corner1[x], corner2[x]], [corner1[y], corner2[y]], [corner1[z], corner2[z]],
                    line_width=2, color=clr, figure=self.figure)

    def draw_text(self, x, y, z, text, color):
        mlab.text3d(0,0,0, text, scale=0.3, color=color, figure=self.figure)
    
    def get_box_color(self, class_id):
        if type(class_id) == str:
            class_id = class_name_to_label(class_id)

        colors = [
            (0,1,0),
            (0,0,1),
            (0,1,1),
        ]

        return colors[class_id]

    def to_numpy(self, pointcloud):
        if not isinstance(pointcloud, np.ndarray):
            return pointcloud.cpu().numpy()
        return pointcloud


# KITTI = KittiDataset()
# _, pointcloud, labels, calib = KITTI[37]
# print(pointcloud.shape)

# pred = [{'pred_boxes': tensor([[19.4029,  0.4326, -0.6832,  3.2483,  1.6385,  1.5225,  6.3273],
#         [24.7294,  3.4589, -0.7314,  4.2589,  1.5571,  1.3714,  6.3081],
#         [ 0.1990,  7.8677, -0.9895,  3.6358,  1.5586,  1.4417,  6.0938],
#         [31.6080,  3.6406, -0.6025,  4.2100,  1.6102,  1.4592,  6.2861],
#         [38.6979,  3.6993, -0.5599,  3.6221,  1.5853,  1.5489,  6.2282],
#         [46.5156,  4.0235, -0.3655,  3.8081,  1.5840,  1.5392,  3.2865],
#         [55.6337, -2.2692, -0.2329,  3.6593,  1.6180,  1.5219,  6.2427],
#         [28.3043,  4.0533, -0.7925,  0.4021,  0.5918,  1.4803,  6.5827],
#         [40.4169, -1.7630, -0.3623,  4.1616,  1.5592,  1.5862,  3.2073],
#         [ 2.2620, -4.7594, -1.0531,  0.6414,  0.6154,  1.4674,  1.2008],
#         [25.2737, -1.4847, -0.6597,  1.5603,  0.3929,  1.5712,  6.2296],
#         [ 3.1774, -5.3730, -0.8268,  0.7852,  0.6185,  1.8156,  4.2028],
#         [17.1161,  4.7242, -0.9604,  4.2199,  1.6030,  1.3866,  2.9143]],
#        device='cuda:0'), 'pred_scores': tensor([0.9641, 0.7366, 0.5543, 0.4312, 0.2171, 0.2127, 0.1781, 0.1690, 0.1525,
#         0.1475, 0.1276, 0.1198, 0.1106], device='cuda:0'), 'pred_labels': tensor([1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 2, 1], device='cuda:0')}]


# visualizer = KittiVisualizer()
# objects = model_output_to_kitti_objects(pred)
# visualizer.visualize(pointcloud, objects)
# visualizer.show()
# visualizer.visualize(pointcloud, objects, labels, calib)
# visualizer.visualize(pointcloud, labels, calib=calib)


 

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