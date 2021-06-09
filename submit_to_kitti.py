import pickle
import numpy as np
import os
import math

from visualization.KittiUtils import KittiCalibration, label_to_class_name

def prcoess3D(center, h, w, l, r_y):
    Rmat = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                    [0, 1, 0],
                    [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners])
    corners_3D = np.dot(Rmat, corners) + center

    return corners_3D.T


# Testing data dir
data_dir = "data/kitti/testing/"
calib_dir = data_dir + "calib/"
label_2_dir = data_dir + "label2/"

if not os.path.exists(label_2_dir):
    os.makedirs(label_2_dir)

img_ids = []
img_names = sorted(os.listdir(calib_dir))

for img_name in img_names:
    img_id = img_name.split(".txt")[0]
    img_ids.append(img_id)

# Load model's Pickle file
with open("object.pkl", "rb") as file:
    testing_objects = pickle.load(file)

for img_id in img_ids:
    print(img_id)
    img_label_file_path = label_2_dir + img_id + ".txt"
    calib_obj = KittiCalibration(calib_dir + img_id + ".txt")

    with open(img_label_file_path, "w") as img_label_file:
        bbox_dicts = testing_objects[int(img_id)]
        for bbox_dict in bbox_dicts:
            x = bbox_dict.bbox_3d.x
            y = bbox_dict.bbox_3d.y
            z = bbox_dict.bbox_3d.z
            h = bbox_dict.bbox_3d.height
            w = bbox_dict.bbox_3d.width
            l = bbox_dict.bbox_3d.length
            r_y = bbox_dict.bbox_3d.rotation

            center = np.array([[x], [y], [z]])
            points_3D = prcoess3D(center, h, w, l, r_y)
            corners_2D = calib_obj.project_lidar_to_image(points_3D)

            minimum = corners_2D.min(axis=0)
            maximum = corners_2D.max(axis=0)

            left = minimum[0]
            top = minimum[1]
            right = maximum[0]
            bottom = maximum[1]

            label = label_to_class_name(bbox_dict.label)
            score = bbox_dict.score * 100

            # (type, truncated, occluded, alpha, left, top, right, bottom, h, w, l, x, y, z, ry, score)
            img_label_file.write("%s -1 -1 -10 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % \
                                 (label, left, top, right, bottom, h, w, l, x, y, z, r_y, score))