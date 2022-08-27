class Config:
  def __init__(self):
    pass
  
  networks = [
    "yolo3_mobilenet1.0_coco",
    "ssd_512_resnet50_v1_coco",
    "center_net_resnet101_v1b_dcnv2_coco",
    "yolo3_darknet53_coco",
    "faster_rcnn_fpn_syncbn_resnest269_coco"
  ]