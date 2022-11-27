# Author(s): Lucas Hirt
# Date modified: 11/27/2022
# This program utilizes the comp viz package and serves as not only an example of a use case of comp viz
# but in itself serves as a useful tool to perform object detection tasks for images provided by the user.
# From the outputs of this program, not only can object detection image results be observed, but data
# can be extrapolated from the results to measure data like network performace, like precision and speed.
# Functionality
#   - User can bring their own images and have the object detector scan for objects in the image
#   - User can bring multiple images (potentially thousands!)
#   - User can save results and calculate performance metrics from the data.
#   - User can change the network and objects they are detecting for in realtime!

import os
import sys
import json
import pathlib

import comp_viz
import client_helper

if __name__ == "__main__":
  print("===========================================")
  print("Welcome to Lucas' computer vision detector!")
  print("===========================================")

  # get all available computer vision models from comp_viz package
  available_models = comp_viz.utils.ObjectDetection.get_networks()

  # show all models and acquire user's choice
  model_name = client_helper.get_model_choice(available_models)

  # instantiate the detection model
  model = comp_viz.object_detection.Model(model_name)
  print(f"{model_name} initialized!")

  print("-------------------------------------------")

  # options for user in the while loop, and termination condition
  options = ["Choose a different model","Predict image(s).","Quit"]
  termination_char = str(len(options) - 1)
  input_char = -1

  # enter loop for user to make predictions
  while input_char != termination_char:
    print("Please make a selection:")
    client_helper.print_enum(options)
    input_char = input("> ")
    
    # user wants to pick a new model
    if input_char == str(0):
      model_name = client_helper.get_model_choice(available_models)
      model = comp_viz.object_detection.Model(model_name)
      continue

    # user wants to predict with chosen model and an image
    elif input_char == str(1):
      while 1:
        print("Please enter the object classes to perform inference on. (Comma separated)")
        object_classes = input("Ex: bicycle, cell phone, chair ... > ")
        object_classes = comp_viz.utils.ObjectDetection.format_object_classes(object_classes.split(","))
        if client_helper.is_valid_object_classes(model, object_classes):
          model.set_classes(object_classes)
          print(f"Object classes: {model.get_classes()} initialized.")
          break

      # prepare output directory
      if not os.path.exists("inference"):
        os.mkdir("inference")
      print("Please input a unique name for this run.")
      dir_path = input("> ")
      out_dir_path = f"inference/{dir_path}"
      if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)
      print(f"Results saved in directory: {out_dir_path}")

      # determine if boundbox images should be produced for each image inferred
      while 1:
        print("Create image with bounding box for each inference performed? (y/n).")
        print("(WARNING: Will SIGNIFICANTLY increase the time required to complete operation if y is chosen.")
        produce_images = input("> ")
        produce_images = produce_images.lower()
        if produce_images == 'y' or produce_images == 'n':
          if produce_images == 'y':
            out_dir_path_images = os.path.join(out_dir_path,"images")
            if not os.path.exists(out_dir_path_images):
              os.mkdir(out_dir_path_images)
          else:
            produce_images = False
          break
        print(f"Invalid input: {produce_images}.")
      
      # get image path and perform inference
      print("Please enter the path to your image(s).")
      in_path = input("> ")
      if os.path.isdir(in_path):
        images = client_helper.get_dir_images(in_path)
      else:
        images = [client_helper.get_image(in_path)]
      print(f"Image count: {len(images)}.")
      for img_fname in images:
        if produce_images:
          img, pred = model.get_image_prediction(img_fname,.5)
          comp_viz.utils.Tools.save_image(img,os.path.join(out_dir_path_images,os.path.basename(img_fname)))
        else:
          pred = model.get_prediction(img_fname)
        with open(f"{os.path.join(out_dir_path,pathlib.Path(img_fname).stem)}.txt", "w") as f:
          f.write(json.dumps(pred,indent=2))
      print(f"Complete! Check directory: {out_dir_path}")
        
    # user wants to end program
    elif input_char == termination_char:
      print("Goodbye!") 
      sys.exit(0)
    
    # input is not valid
    else:
      print(f"Input: {input_char} is not valid. Please try again.")
      continue






  


  