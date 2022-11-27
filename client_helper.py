# Author(s): Lucas Hirt
# Date modified: 11/27/2022
# This program serves as a collection of helper functions for the primary user interface (client.py).
# Key functionality this program provides to the primary interface are:
#   - Handling model choice made by user
#   - Ensuring validity of inputs made by user- throwing appropriate messages if input is erroneous.
#   - Output functions to make client program more readable
#   - Help flag functions to enable user to get more information
#   - Getter and setter functions for image retrieval.

import comp_viz
import os

def get_model_choice(models: list) -> str:
  print("Please choose from the following models:")
  print("Models are in order from fastest (lowest number) to most precise (highest num).")
  print_enum(models)
  print("To see information about a model, enter it's corresponding number followed by the letter d. Ex: 0d or 1d")
  user_input = input("> ")
  valid_inputs = [str(num) for num in range(len(models))] + [str(num) + 'd' for num in range(len(models))]
  if not is_valid_input(user_input,valid_inputs):
    print(f"Error: choice {user_input} was not a valid input. Please try again.")
    return get_model_choice(models)
  return handle_model_choice_input(user_input,models)


def print_enum(list_vals: list):
  for i, val in enumerate(list_vals):
    print(f"{i}: {val}")


def is_valid_input(x: str, valid_x: list) -> bool:
  if x not in valid_x:
    return False
  return True


def handle_model_choice_input(user_input: str, models: list) -> str:
  model = models[int(user_input[0])]
  if "d" in user_input:
    print(f"{describe_model(model)[0]}: {describe_model(model)[1]}")
    input("Press any key to continue...")
    return get_model_choice(models)
  return model


def describe_model(name: str) -> tuple:
  model = {}
  model["yolo3_mobilenet1.0_coco"] = "As fast as it gets. Useful for real-time object detection. Comes with a cost of decreased precision."
  model["center_net_resnet101_v1b_dcnv2_coco"] = "Runner up for fastest network. Useful if one can afford sacrifing a bit of speed for improved precision."
  model["yolo3_darknet53_coco"] = "Gold standard for balance between precision and speed. Commonly used in real-time inference applications."
  model["faster_rcnn_fpn_syncbn_resnest269_coco"] = "As precise as it gets. Comes with a large cost in speed. Not recommended for real-time inference."
  model["ssd_512_resnet50_v1_coco"]  = "Third fastest network. Slightly faster than YOLO3 darkent, but comes with a cost in accuracy."
  if name not in model:
    return name, "No model description available at this time."
  return name, model[name]

def is_valid_object_classes(model, object_classes) -> bool:
  model_classes = model.get_classes()
  for object_class in object_classes:
    if object_class not in model_classes:
      print(f"Object class \"{object_class}\" is not available for chosen model.")
      return False
  return True

def get_dir_images(dir):
  valid_extensions = ["jpg","png","jpeg"]
  dir_contents = os.listdir(dir)
  image_paths = []
  i = 0
  while i < len(dir_contents):
    file_path = dir_contents[i]
    if file_path.split(".")[-1] in valid_extensions:
      image_paths.append(os.path.join(dir,file_path))
    i += 1
  return image_paths

def get_image(path):
  valid_extensions = ["jpg","png","jpeg"]
  fname = os.path.basename(path)
  if fname.split(".")[-1] in valid_extensions:
    return path