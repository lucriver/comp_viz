import mxnet
import gluoncv
import numpy
from .. import utils

class Model:
  def __init__(self,network_name=None):
    self.net_name = network_name
    self.net = gluoncv.model_zoo.get_model(network_name, pretrained=True)

  def predict(self,fname):
    utils.verify_exists(fname)
    img = mxnet.image.imread(fname)
    x, frame = self._prepare_image(img)
    cids, scores, bboxes = self.net(x)
    return (cids[0],scores[0],bboxes[0])

  def _prepare_image(self,image):
    if "yolo" in self.net_name:
      return gluoncv.data.transforms.presets.yolo.transform_test(image,short=512)
    elif "rcnn" in self.net_name:
      return gluoncv.data.rcnn.presets.yolo.transform_test(image,short=512)
    elif "ssd" in self.net_name:
      return gluoncv.data.transforms.presets.ssd.transform_test(image,short=512)

