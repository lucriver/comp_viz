"""Binary tree traversal.
Routines
--------
inorder_traverse(tree: `SupportedTreeType`, recursive: `bool`)
    Perform in-order traversal.
"""

import os

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

  def _get_networks():
    return obj_det_config.networks


class Tools:
  def verify_exists(fname):
    if not Tool._exists(fname):
      print(f"Error: file {fname} could not be located.")
      return

  def exists(fname):
    return Tool._exists(fname)
    
  def _exists(fname):
    if os.path.exists(fname):
      return True
    return False

