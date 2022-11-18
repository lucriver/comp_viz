import mxnet
import gluoncv
import numpy
import time

from .. import utils

class Model:
  def __init__(self,network_name=None):
    self.net_name = network_name
    self.net = gluoncv.model_zoo.get_model(network_name, pretrained=True)

  def get_prediction(self,fname,nms=0) -> dict:
    utils.Tools.verify_exists(fname)
    start = time.time()
    cids, scores, bboxes = self.predict(fname,nms)
    end = time.time()
    prediction = {}
    prediction["image"] = str(fname)
    prediction['class_ids'] = cids
    prediction['confidence_scores'] = scores
    prediction['bounding_boxes'] = bboxes
    prediction['nms_thresh'] = nms
    prediction['class_map'] = [{cid: self.get_classes()[cid]} for cid in set(cids)]
    prediction['time'] = round(float(end - start),4)
    return prediction

  def list_classes(self):
    print(self.net.classes)

  def get_classes(self):
    return self.net.classes

  def predict(self,fname,nms) -> tuple:
    img = mxnet.image.imread(fname)
    x, _ = self.__prepare_image(img)
    pred = self.net(x)
    cids, scores, bboxes = self._extract_cids_scores_bboxes(pred)
    if nms == 0:
      return (cids,scores,bboxes)
    nms_cids, nms_scores, nms_bboxes = self._apply_nms(cids, scores, bboxes, nms)
    return (nms_cids,nms_scores,nms_bboxes)

  def set_object_classes(self,object_classes):
    self.net.reset_class(object_classes, reuse_weights=object_classes)

  def _extract_cids_scores_bboxes(self,pred):
    cids = self._get_class_ids(pred[0])
    scores = self._get_scores(pred[1])
    bboxes = self._get_bboxes(pred[2])
    return (cids, scores, bboxes)

  def _apply_nms(self,cids: list, scores: list, bboxes: list, nms: float):
    prune_indexes = []
    for i, score in enumerate(scores):
      if score < nms:
        prune_indexes.append(i)
    for index in prune_indexes[::-1]:
      cids.pop(index)
      scores.pop(index)
      bboxes.pop(index)
    return (cids, scores, bboxes)

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
        confidence_scores.append(round(float(nparray[0]),3))
    return confidence_scores

  def _get_bboxes(self,bboxes):
    bounding_boxes = []
    for ndarray in bboxes[0]:
      nparray = ndarray.asnumpy()
      if nparray[0] != -1:
        bb = [int(corner) for corner in nparray.tolist()]
        bounding_boxes.append(bb)
    return bounding_boxes

  def __prepare_image(self,image):
    if "yolo" in self.net_name:
      return gluoncv.data.transforms.presets.yolo.transform_test(image,short=512)
    elif "rcnn" in self.net_name:
      return gluoncv.data.rcnn.presets.yolo.transform_test(image,short=512)
    elif "ssd" in self.net_name:
      return gluoncv.data.transforms.presets.ssd.transform_test(image,short=512)
