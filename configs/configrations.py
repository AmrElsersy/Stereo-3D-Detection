class Configration:
    path_models = 'configs/checkpoint/3D_Detection/'
    ext_models = '.pth'
    models = {"PV-RCNN": "%spv_rcnn_8369%s"% (path_models,ext_models),
              "PointPillars": "%spointpillar_7728%s"% (path_models,ext_models),
              "Second": "%ssecond_7862%s"% (path_models,ext_models),
              "PointRCNN": "%spointrcnn_7870%s"% (path_models,ext_models),
              "PointRCNNIoU": "%spointrcnn_iou_7875%s"% (path_models,ext_models),
              "PartFree": "%sPartA2_free_7872%s"% (path_models,ext_models),
              "PartAnchor": "%sPartA2_7940%s"% (path_models,ext_models)}

    path_cfg = 'configs/cfgs/kitti_models/'
    ext_cfg = '.yaml'
    cfg = {"PV-RCNN": "%spv_rcnn%s"% (path_cfg,ext_cfg),
           "PointPillars": "%spointpillar%s"% (path_cfg,ext_cfg),
           "Second": "%ssecond%s"% (path_cfg,ext_cfg),
           "PointRCNN": "%spointrcnn%s"% (path_cfg,ext_cfg),
           "PointRCNNIoU": "%spointrcnn_iou%s"% (path_cfg,ext_cfg),
           "PartFree": "%sPartA2_free%s"% (path_cfg,ext_cfg),
           "PartAnchor": "%sPartA2%s"% (path_cfg,ext_cfg)}

class PVRCNN:
    type = "PV-RCNN"
    config = Configration()
    model = config.models["PV-RCNN"]
    cfg = config.cfg["PV-RCNN"]

class PointPillars:
    type = "PointPillars"
    config = Configration()
    model = config.models["PointPillars"]
    cfg = config.cfg["PointPillars"]

class Second:
    type = "Second"
    config = Configration()
    model = config.models["Second"]
    cfg = config.cfg["Second"]

class PointRCNN:
    type = "PointRCNN"
    config = Configration()
    model = config.models["PointRCNN"]
    cfg = config.cfg["PointRCNN"]

class PointRCNNIoU:
    type = "PointRCNNIoU"
    config = Configration()
    model = config.models["PointRCNNIoU"]
    cfg = config.cfg["PointRCNNIoU"]

class PartFree:
    type = "PartFree"
    config = Configration()
    model = config.models["PartFree"]
    cfg = config.cfg["PartFree"]

class PartAnchor:
    type = "PartAnchor"
    config = Configration()
    model = config.models["PartAnchor"]
    cfg = config.cfg["PartAnchor"]