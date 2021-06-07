import pickle
import numpy as np
import os
import math
import sys

from visualization.KittiUtils import KittiCalibration, KittiObject

def ProjectTo2Dbbox(center, h, w, l, r_y, P2):
    # input: 3Dbbox in (rectified) camera coords

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    points = np.array([p0, p1, p2, p3, p4, p5, p6, p7])

    points_hom = np.ones((points.shape[0], 4)) # (shape: (8, 4))
    points_hom[:, 0:3] = points

    # project the points onto the image plane (homogeneous coords):
    img_points_hom = np.dot(P2, points_hom.T).T # (shape: (8, 3)) (points_hom.T has shape (4, 8))
    # normalize:
    img_points = np.zeros((img_points_hom.shape[0], 2)) # (shape: (8, 2))
    img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
    img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

    u_min = np.min(img_points[:, 0])
    v_min = np.min(img_points[:, 1])
    u_max = np.max(img_points[:, 0])
    v_max = np.max(img_points[:, 1])

    left = int(u_min)
    top = int(v_min)
    right = int(u_max)
    bottom = int(v_max)

    projected_2Dbbox = [left, top, right, bottom]

    return projected_2Dbbox

experiment_name = "anynet_sfa_submission" 

# Testing data dir
data_dir = "data/kitti/testing/"
calib_dir = data_dir + "calib/"

results_dir = "results/"

experiment_results_dir = results_dir + experiment_name + "/"
results_data_dir = experiment_results_dir + "data/"

if os.path.exists(experiment_results_dir):
    raise Exception("That experiment name already exists!")
else:
    os.makedirs(experiment_results_dir)
    os.makedirs(results_data_dir)


img_ids = []
img_names = os.listdir(calib_dir)

for img_name in img_names:
    img_id = img_name.split(".txt")[0]
    img_ids.append(img_id)

# Load model's Pickle file
with open("", "rb") as file: 
    eval_dict = pickle.load(file)

calib_obj = KittiCalibration(calib_dir)
for img_id in img_ids:
    print(img_id)

    img_label_file_path = results_data_dir + img_id + ".txt"

    with open(img_label_file_path, "w") as img_label_file:
        P2 = calib_obj.calib_matrix["P2"]
        bbox_dicts = eval_dict[img_id]
        
        if img_id in eval_dict: # (if any predicted bboxes for the image:) (otherwise, just create an empty file)
            calib = calibread(calib_dir + img_id + ".txt")
            P2 = calib["P2"]

            bbox_dicts = eval_dict[img_id]
            for bbox_dict in bbox_dicts:
                pred_center_BboxNet = bbox_dict["pred_center_BboxNet"]
                pred_x = pred_center_BboxNet[0]
                pred_y = pred_center_BboxNet[1]
                pred_z = pred_center_BboxNet[2]
                pred_h = bbox_dict["pred_h"]
                pred_w = bbox_dict["pred_w"]
                pred_l = bbox_dict["pred_l"]
                pred_r_y = bbox_dict["pred_r_y"]

                projected_2Dbbox = ProjectTo2Dbbox(pred_center_BboxNet, pred_h, pred_w, pred_l, pred_r_y, P2)
                left = projected_2Dbbox[0]
                top = projected_2Dbbox[1]
                right = projected_2Dbbox[2]
                bottom = projected_2Dbbox[3]

                score = bbox_dict["score_2d"]

                # (type, truncated, occluded, alpha, left, top, right, bottom, h, w, l, x, y, z, ry, score)
                img_label_file.write("Car -1 -1 -10 %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % \
                    (left, top, right, bottom, pred_h, pred_w, pred_l, pred_x, pred_y, pred_z, pred_r_y, score))