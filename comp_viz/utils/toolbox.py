"""Binary tree traversal.
Routines
--------
inorder_traverse(tree: `SupportedTreeType`, recursive: `bool`)
    Perform in-order traversal.
"""

import os
import mxnet
import gluoncv
import numpy
import cv2

from ..config import Models as models_config
from ..config import ObjectDetection as obj_det_config

class Models:
  def list_tasks():
    print(Models._get_tasks())

  def get_tasks():
    return Models._get_tasks()

  def _get_tasks():
    return Models.models_config.tasks


class ObjectDetection:
  def list_networks():
    print(ObjectDetection._get_networks())

  def get_networks():
    return ObjectDetection._get_networks()

  def get_network_resolution(net_name):
    return ObjectDetection._get_resolution(net_name)

  def format_object_classes(object_classes: list) -> list:
    i = 0
    while i < len(object_classes):
      if object_classes[i] == "":
        object_classes.pop(i)
        continue
      object_classes[i] = object_classes[i].lower().strip()
      i += 1
    return object_classes

  def resize_bbox(bbox: list, orig: tuple, dest: tuple):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    x_scale = dest[1] / orig[1]
    y_scale = dest[0] / orig[0]
    return [float(numpy.round(x_min*x_scale)), 
            float(numpy.round(y_min*y_scale)),
            float(numpy.round(x_max*x_scale)),
            float(numpy.round(y_max*y_scale))]

  def show_pred_bboxes_image(img_fname: str, bboxes: list, labels = [], class_names = [], scores = []):
    img = ObjectDetection.get_pred_bboxes_image(img_fname,bboxes,labels,class_names,scores)
    gluoncv.utils.viz.plot_image(img)

  def get_pred_bboxes_image(img_fname: str, bboxes: list, labels = [], class_names = [], scores = []):
    img = Tools.get_mxnet_image(img_fname)
    return gluoncv.utils.viz.cv_plot_bbox(img,numpy.array(bboxes),labels=numpy.array(labels),scores=numpy.array(scores),class_names=class_names,thresh=0.)

  def _get_networks():
    return [network for network in obj_det_config.networks.keys()]

  def _get_resolution(net_name):
    return obj_det_config.networks[net_name]["resolution"]

class Tools:
  def verify_exists(fname: str):
    if not Tools._exists(fname):
      print(f"Error: file {fname} could not be located.")
      return

  def exists(fname: str) -> bool:
    return Tools._exists(fname)

  def get_mxnet_image(fname: str):
    Tools.verify_exists(fname)
    return mxnet.image.imread(fname)

  def show_image(img: numpy.ndarray):
    gluoncv.utils.viz.plot_image(img)

  def filename_show_image(fname: str):
    Tools.verify_exists(fname)
    img = mxnet.image.imread(fname)
    gluoncv.utils.viz.plot_image(img)

  def save_image(img: numpy.ndarray, path: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
    
  def _exists(fname: str) -> bool:
    if os.path.exists(fname):
      return True
    return False

