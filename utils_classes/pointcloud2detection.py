from visualization.KittiDataset import KittiDataset
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiUtils import *

def predict_pseudo_lidar(model):
    visualizer = KittiVisualizer()
    KITTI = KittiDataset('data/kitti/training')
    root = "/home/amrelsersy/SFA3D/dataset/kitti/testing/pseudo_SDN"
    paths = os.listdir(root)

    for path in paths:
        path = root + "/" + path
        pointcloud = KITTI.read_pointcloud_bin(path)
        image = KITTI.read_image_cv2(path.replace("pseudo_SDN", "image_2").replace("bin", "png"))
        calib = KittiCalibration(path.replace("pseudo_SDN", "calib").replace("bin", "txt"))
        print(pointcloud.shape, image.shape)

        pred = model.predict(pointcloud)
        objects = model_output_to_kitti_objects(pred)

        visualizer.visualize_scene_2D(pointcloud, image, objects, calib=calib)
        # visualizer.visualize_scene_bev(pointcloud, objects)
        # visualizer.visualize_scene_3D(pointcloud, objects)
        # visualizer.visualize_scene_image(image, objects, calib)


def predict_lidar(model, index):
    print("========== LIDAR only =============")
    visualizer = KittiVisualizer()
    KITTI = KittiDataset('data/kitti/training')

    # for i in range(8, 100):
    image, pointcloud, labels, calib = KITTI[index]
    print(pointcloud.shape)

    pred = model.predict(pointcloud)

    objects = model_output_to_kitti_objects(pred)
    visualizer.visualize_scene_2D(pointcloud, image, objects, calib=calib)

    # visualizer.visualize_scene_3D(pointcloud, objects, labels, calib)
    # visualizer.visualize_scene_bev(pointcloud, objects, calib=calib)
    # visualizer.visualize_scene_image(image, objects, calib)
