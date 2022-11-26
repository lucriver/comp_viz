import os
import mxnet
import gluoncv
import numpy
import cv2

from ..config import Models as models_config
from ..config import ObjectDetection as obj_det_config

class Models:
  """Utility class centered around conveying available functionality for the comp_viz package. 
  """

  def list_tasks():
    """Show available tasks for the comp_viz package. 
    """
    print(Models._get_tasks())

  def get_tasks():
    """Get available tasks for the comp_viz package. 
    
    :rtype: list
    """
    return Models._get_tasks()

  def _get_tasks():
    return models_config.tasks


class ObjectDetection:
  """Utility class centered around object detection tasks relevant but not limited to the comp_viz object detection package.
  """

  def list_networks():
    """Show list of the available networks that can be used with the comp_viz object_detection sub-package.
    """
    print(ObjectDetection._get_networks())

  def get_networks():
    """Get list of the available networks that can be used with the comp_viz object_detection sub-package.

    :rtype: List
    """
    return ObjectDetection._get_networks()

  def get_network_resolution(net_name):
    """Get image inference resolution of the specified network.

    :param net_name: A valid network name among the results in get_networks() or list_networks() method.
    :type net_name: string
    :rtype: int
    """
    return ObjectDetection._get_resolution(net_name)

  def format_object_classes(object_classes: list) -> list:
    """Given a list of object classes, format all the elements such that they are readable by the network.

    :param object_classes: List of object classes.
    :type object_classes: list
    :rtype: list
    """
    i = 0
    while i < len(object_classes):
      if object_classes[i] == "":
        object_classes.pop(i)
        continue
      object_classes[i] = object_classes[i].lower().strip()
      i += 1
    return object_classes

  def resize_bbox(bbox: list, orig: tuple, dest: tuple):
    """Given a bounding box of the format [x,min, y_min, x_max, y_max] and the original image resolution, return a new bounding box resized to the desired image size.

    :param bbox: Bounding box of the form [x_min, y_min, x_max, y_max].
    :type bbox: list
    :param orig: Original image resolution of form (height, width, shape). Ex. (500,800,3)
    :type orig: Tuple or ndarray.shape
    :param dest: Image to resize resolution of form (height, width, shape). Ex. (600,900,3)
    :type dest: Tuple or ndarray.shape
    :rtype: list
    """
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    x_scale = dest[1] / orig[1]
    y_scale = dest[0] / orig[0]
    return [float(numpy.round(x_min*x_scale)), 
            float(numpy.round(y_min*y_scale)),
            float(numpy.round(x_max*x_scale)),
            float(numpy.round(y_max*y_scale))]

  def show_pred_bboxes_image(img_fname: str, bboxes: list, labels = [], class_names = [], scores = []):
    """Given an image and detection bounding box features, plot the bounding box to the image and show it.

    :param img_fname: Path to image to plot bounding box to.
    :type img_fname: string
    :param bboxes: Bounding boxes of form [[x_min,y_min,x_max,y_max],...] to plot to the image/
    :type bboxes: List[List]
    :param labels: Class id values to mape to each bounding box and class name.
    :type labels: List[int]
    :param class_names: List of object classes:
    :type class_names: List[string]
    :param scores: List of confidence values for the bounding boxes.
    :type scores: List[float]
    :rtype: void
    """
    img = ObjectDetection.get_pred_bboxes_image(img_fname,bboxes,labels,class_names,scores)
    Tools.show_image(img)

  def get_pred_bboxes_image(img_fname: str, bboxes: list, labels = [], class_names = [], scores = []):
    """Given an image and detection bounding box features, plot the bounding box to the image and return it.

    :param img_fname: Path to image to plot bounding box to.
    :type img_fname: string
    :param bboxes: Bounding boxes of form [[x_min,y_min,x_max,y_max],...] to plot to the image/
    :type bboxes: List[List]
    :param labels: Class id values to mape to each bounding box and class name.
    :type labels: List[int]
    :param class_names: List of object classes:
    :type class_names: List[string]
    :param scores: List of confidence values for the bounding boxes.
    :type scores: List[float]
    :rtype: numpy.ndarray
    """
    img = Tools.get_mxnet_image(img_fname)
    return gluoncv.utils.viz.cv_plot_bbox(img,numpy.array(bboxes),labels=numpy.array(labels),scores=numpy.array(scores),class_names=class_names,thresh=0.)

  def _get_networks():
    return [network for network in obj_det_config.networks.keys()]

  def _get_resolution(net_name):
    return obj_det_config.networks[net_name]["resolution"]

class Tools:
  """Utility class centered around images and filesnames.
  """
  def verify_exists(fname: str):
    if not Tools._exists(fname):
      print(f"Error: file {fname} could not be located.")
      return

  def exists(fname: str) -> bool:
    """Boolean function to determine if path to filename exists.

    :param fname: Path to file.
    :type fname: string
    :rtype: boolean
    """
    return Tools._exists(fname)

  def get_mxnet_image(fname: str):
    """Given path to an image file, return the said image in the form of an mxnet ndarray.

    :param fname: Path to file.
    :type fname: string
    :rtype: mxnet.ndarray.ndarray.NDArray
    """
    Tools.verify_exists(fname)
    return mxnet.image.imread(fname)

  def get_cv2_image(fname: str):
    """Given path to an image file, return the said image in the form of an numpy ndarray using openCV.

    :param fname: Path to file.
    :type fname: string
    :rtype: numpy.ndarray
    """
    Tools.verify_exists(fname)
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

  def show_image(img: numpy.ndarray):
    """Given an image in the form of an numpy ndarray, show the image to the screen.

    :param img: Image in the form of an ndarray.
    :type img: numpy.ndarray or mxnet.ndarray.ndarray.NDArray
    :rtype: void
    """
    gluoncv.utils.viz.plot_image(img)

  def filename_show_image(fname: str):
    """Given path to an image file, show the said image file to the screen.

    :param fname: Path to image.
    :type fname: string
    :rtype: void
    """
    Tools.verify_exists(fname)
    img = mxnet.image.imread(fname)
    gluoncv.utils.viz.plot_image(img)

  def save_image(img: numpy.ndarray, path: str):
    """Given an image in the form of an ndarray, save it to the path specified.

    :param img: Image in the form of an ndarray.
    :type img: numpy.ndarray or mxnet.ndarray.ndarray.NDArray
    :param path: Path to save image to.
    :type path: string
    :rtype: void
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
    
  def _exists(fname: str) -> bool:
    if os.path.exists(fname):
      return True
    return False

