import argparse
import os
import json
import mxnet as mx
import gluoncv as gcv
from .config import Config

class CompViz:
  def __init__(self):
    pass

  @classmethod
  def list_networks(cls):
    networks = cls.__get_networks()
    print(json.dumps(networks,indent=2))
    
  @classmethod
  def get_networks(cls):
    networks = cls.__get_networks()
    return networks
  
  @classmethod
  def __get_networks(cls,network_list=Config.networks):
    networks = network_list
    return networks
    
  @classmethod
  def __verify_file_exists(cls,fname):
    if not cls.file_exists(fname):
      print(f"Error: File {fname} could not be located.")
      return

  @classmethod
  def __file_exists(cls,fname):
    if os.path.exists(fname):
      return True
    return False


class CompVizModel(CompViz):
  def __init__(self):
    pass
  

