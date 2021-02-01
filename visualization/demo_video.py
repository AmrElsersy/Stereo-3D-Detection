import os
import cv2
import numpy as np

ROOT_DIR = os.path.dirname('/home/ayman/FOE-Linux/Graduation_Project/KITTI/')

class KittiVideo:
    """ Load data for KITTI videos """

    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = self.read_calib_from_video(calib_dir)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted(
            [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        )
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )

        assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = self.img_filenames[idx]
        return cv2.imread(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = self.lidar_filenames[idx]
        # return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib

    def read_calib_from_video(self, calib_root_dir):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        data = {}
        cam2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_cam_to_cam.txt")
        )
        velo2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_velo_to_cam.txt")
        )
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
        data["R0_rect"] = cam2cam["R_rect_00"]
        data["P2"] = cam2cam["P_rect_02"]
        return data

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, "2011_09_26_drive_0001")

    dataset = KittiVideo(
        img_dir=os.path.join(video_path, "2011_09_26_drive_0001_sync/2011_09_26/image_02/data"),
        lidar_dir=os.path.join(video_path, "2011_09_26_drive_0001_sync/2011_09_26/velodyne_points/data"),
        calib_dir=os.path.join(video_path, "2011_09_26_calib/2011_09_26")
    )

    print("Loaded Video Frames: ",len(dataset))
    height, width, channels = dataset.get_image(0).shape    

    outVideo = cv2.VideoWriter('demo_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (width, height))

    for i in range(len(dataset)):
        img = dataset.get_image(i)
        pc = dataset.get_lidar(i)
        outVideo.write(dataset.get_image(i))
            

viz_kitti_video()