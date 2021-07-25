import pickle
import os
import math

from visualization.KittiUtils import KittiCalibration, label_to_class_name
from visualization.KittiVisualization import KittiVisualizer

def warpToPi(angle):
    angle = angle % (2*math.pi)
    if angle > math.pi:
        angle -= 2*math.pi
    elif angle < -math.pi:
        angle -= 2 * math.pi

    return angle

# Testing data dir
data_dir = "data/kitti/testing/"
calib_dir = data_dir + "calib/"
label_2_dir = data_dir + "label2/"

if not os.path.exists(label_2_dir):
    os.makedirs(label_2_dir)

img_ids = []
img_names = sorted(os.listdir(calib_dir))

vis = KittiVisualizer()

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
            r_y = warpToPi(r_y)
            alpha = r_y + math.atan2(z, x) + 1.5*math.pi
            alpha = warpToPi(alpha)

            corners = vis.convert_3d_bbox_to_corners(bbox_dict.bbox_3d, calib_obj)
            proj_corners = calib_obj.project_lidar_to_image(corners)
            filtered_corners = vis.filter_truncated_points(proj_corners)

            minimum = filtered_corners.min(axis=0)
            maximum = filtered_corners.max(axis=0)

            left = minimum[0]
            top = minimum[1]
            right = maximum[0]
            bottom = maximum[1]

            label = label_to_class_name(bbox_dict.label)
            score = bbox_dict.score * 100

            # (type, truncated, occluded, alpha, left, top, right, bottom, h, w, l, x, y, z, ry, score)
            img_label_file.write("%s -1 -1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % \
                                 (label, alpha, left, top, right, bottom, h, w, l, x, y, z, r_y, score))