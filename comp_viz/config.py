class CompViz:
  """Most parental configuration class for the comp_viz package.
  
  :ivar version: Version number for the comp_viz package.
  """
  version = "1.0.0"

class Models(CompViz):
  """Configuration class for available functionality for the comp_viz package. 

  :ivar tasks: List of supported tasks provided by comp_viz package.
  """
  tasks = ["Object Detection"]

class ObjectDetection(CompViz):
  """Configuration class for the object detection task for the comp_viz package.

  :ivar networks: Dictionary of supported networks for object detection for the comp viz package. Each network has an associated inference resolution.
  """
  networks = {
      "yolo3_mobilenet1.0_coco": { "resolution": 416 },
      "yolo3_darknet53_coco": { "resolution": 416 },
      "ssd_512_resnet50_v1_coco": { "resolution": 416 },
      "center_net_resnet101_v1b_dcnv2_coco": { "resolution": 416 },
      "faster_rcnn_fpn_resnet50_v1b_coco": { "resolution": 416 },
      "faster_rcnn_fpn_syncbn_resnest269_coco": { "resolution": 416 }
  }