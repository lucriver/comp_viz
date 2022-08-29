import mxnet
import gluoncv
import numpy

from .. import utils

class Model:
  def __init__(self,network_name=None):
    self.net_name = network_name
    self.net = gluoncv.model_zoo.get_model(network_name, pretrained=True)

  def get_prediction(self,fname,nms=0.5) -> tuple:
    utils.verify_exists(fname)
    img = mxnet.image.imread(fname)
    x, frame = self.__prepare_image(img)
    pred = self.net(x)
    cids = self._get_class_ids(pred[0])
    scores = self._get_scores(pred[1])
    bboxes = self._get_bboxes(pred[2])
    nms_cids, nms_scores, nms_bboxes = self._apply_nms(cids, scores, bboxes, nms)
    return (nms_cids, nms_scores, nms_bboxes)

  def get_prediction_no_nms(self,fname):
    utils.verify_exists(fname)
    img = mxnet.image.imread(fname)
    x, frame = self.__prepare_image(img)
    pred = self.net(x)
    cids = self._get_class_ids(pred[0])
    scores = self._get_scores(pred[1])
    bboxes = self._get_bboxes(pred[2])
    return (cids, scores, bboxes)

  def list_classes(self):
    print(self.net.classes)

  def get_classes(self):
    return self.net.classes

  def _apply_nms(self,cids: list, scores: list, bboxes: list, nms: float):
    prune_indexes = []
    for i, score in enumerate(scores):
      if score < nms:
        prune_indexes.append(i)
    for index in prune_indexes[::-1]:
      cids.pop(index)
      scores.pop(index)
      bboxes.pop(index)
    return cids, scores, bboxes

  def _get_class_ids(self,cids):
    class_ids = []
    for ndarray in cids[0]:
      nparray = ndarray.asnumpy()
      if nparray[0] != -1:
        class_ids.append(int(nparray[0]))
    return class_ids

  def _get_scores(self,scores):
    confidence_scores = []
    for ndarray in scores[0]:
      nparray = ndarray.asnumpy()
      if nparray[0] != -1:
        confidence_scores.append(float(nparray[0]))
    return confidence_scores

  def _get_bboxes(self,bboxes):
    bounding_boxes = []
    for ndarray in bboxes[0]:
      nparray = ndarray.asnumpy()
      if nparray[0] != -1:
        bb = [int(corner) for corner in nparray.tolist()]
        bounding_boxes.append(bb)
    return bounding_boxes

  def _get_class_ids(self,cids):
    class_ids = []
    for ndarray in cids[0]:
      nparray = ndarray.asnumpy()
      if nparray[0] != -1:
        class_ids.append(int(nparray[0]))
    return class_ids

  def __prepare_image(self,image):
    if "yolo" in self.net_name:
      return gluoncv.data.transforms.presets.yolo.transform_test(image,short=512)
    elif "rcnn" in self.net_name:
      return gluoncv.data.rcnn.presets.yolo.transform_test(image,short=512)
    elif "ssd" in self.net_name:
      return gluoncv.data.transforms.presets.ssd.transform_test(image,short=512)
