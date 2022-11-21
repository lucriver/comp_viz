import mxnet
import gluoncv
import numpy
import time

from .. import utils

class Model:
  def __init__(self,network_name):
    self.net_name = network_name
    self.net = gluoncv.model_zoo.get_model(network_name, pretrained=True)
    self.inference_resolution = utils.ObjectDetection.get_network_resolution(network_name)
    self.default_object_classes = gluoncv.model_zoo.get_model(network_name, pretrained=True).classes

  def list_classes(self):
    print(self._get_classes())

  def get_classes(self):
    return (self._get_classes())

  def get_prediction(self,fname,nms=.5) -> dict:
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
    prediction['class_map'] = {cid: self.get_classes()[cid] for cid in set(cids)}
    prediction['time'] = round(float(end - start),4)
    return prediction

  def get_image_prediction(self,fname,nms=0):
    pred = self.get_prediction(fname)
    pred_img = utils.ObjectDetection.get_pred_bboxes_image(fname,
                                                           pred["bounding_boxes"],
                                                           [key for key in pred["class_map"].keys()],
                                                           [val for val in pred["class_map"].values()],
                                                           pred["confidence_scores"])
    return pred_img, pred

  def show_image_prediction(self,fname,nms=.5):
    image, prediction = self.get_image_prediction(fname)
    utils.Tools.show_image(image)
    print(prediction)

  def predict(self,fname,nms) -> tuple:
    base_img = mxnet.image.imread(fname)
    x, img = self.__prepare_image(base_img)
    pred = self.net(x)
    cids, scores, bboxes = self._extract_cids_scores_bboxes(pred)
    bboxes = [utils.ObjectDetection.resize_bbox(bbox,img.shape,base_img.shape) for bbox in bboxes]
    if nms == 0:
      return (cids,scores,bboxes)
    nms_cids, nms_scores, nms_bboxes = self._apply_nms(cids, scores, bboxes, nms)
    return (nms_cids,nms_scores,nms_bboxes)

  def set_classes(self,object_classes: list):
    unsupported = []
    for obj_class in object_classes:
      if obj_class not in self.default_object_classes:
        unsupported.append(obj_class)
    if unsupported:    
      print(f"WARNING: object classes \"{unsupported}\" are not supported by default for object detection. Expect no capability for detection.")
    self.net.reset_class(object_classes, reuse_weights=object_classes)
    print(f"Complete. Model set to detect for object classes: {self.get_classes()}.")

  def reset_classes(self):
    self.net.reset_class(self.default_object_classes,reuse_weights=self.default_object_classes)
    print("Object classes for detection restored to defaults.")

  def _get_classes(self):
    return self.net.classes
    
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
        bb = [float(corner) for corner in nparray.tolist()]
        bounding_boxes.append(bb)
    return bounding_boxes

  def __prepare_image(self,image):
    if "yolo" in self.net_name:
      return gluoncv.data.transforms.presets.yolo.transform_test(image,short=self.inference_resolution)
    elif "rcnn" in self.net_name:
      return gluoncv.data.transforms.presets.rcnn.transform_test(image,short=self.inference_resolution)
    elif "ssd" in self.net_name:
      return gluoncv.data.transforms.presets.ssd.transform_test(image,short=self.inference_resolution)
    elif "center_net" in self.net_name:
      return gluoncv.data.transforms.presets.center_net.transform_test(image,short=self.inference_resolution)