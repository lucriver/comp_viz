"""Binary tree traversal.
Routines
--------
inorder_traverse(tree: `SupportedTreeType`, recursive: `bool`)
    Perform in-order traversal.
"""

import os

from ..config import networks as available_networks

def list_networks():
  networks = _get_networks()
  print(networks)

def get_networks():
  return _get_networks()

def _get_networks():
  networks = available_networks
  return networks

def verify_exists(fname):
  if not _exists(fname):
    print(f"Error: file {fname} could not be located.")
    return
    
def _exists(fname):
  if os.path.exists(fname):
    return True
  return False

