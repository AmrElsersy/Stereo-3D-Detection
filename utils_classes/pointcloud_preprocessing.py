from pcdet.datasets import DatasetTemplate

class PointcloudPreprocessing(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
    def preprocess_pointcloud(self, pointcloud):
        input_dict = {
            'points': pointcloud,
            'frame_id': 0
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict