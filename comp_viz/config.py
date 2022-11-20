class CompViz:
  version = "0.0.1"

class Models:
  tasks = ["Object Detection"]

class ObjectDetection:
  networks = {
      "yolo3_mobilenet1.0_coco": { "resolution": 320 },
      "yolo3_darknet53_coco": { "resolution": 416 },
      "ssd_512_resnet50_v1_coco": { "resolution": 512 },
      "center_net_resnet101_v1b_dcnv2_coco": { "resolution": 512 },
      "faster_rcnn_fpn_resnet50_v1b_coco": { "resolution": 600 },
      "faster_rcnn_fpn_syncbn_resnest269_coco": { "resolution": 600 }
  }