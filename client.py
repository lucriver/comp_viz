import os
import sys

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
  options = ["Choose a different model","Predict with an image","Quit"]
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
      model = comp_viz.models.Model(model_name)
      continue

    # user wants to predict with chosen model and an image
    elif input_char == str(1):
      if not os.path.exists("inference"):
        os.mkdir("inference")
      print("Please enter the path to your image.")
      image_path = input("> ")
      prediction = model.get_prediction(image_path)
      print(prediction)
      input("Press any button to continue...")

    # user wants to end program
    elif input_char == termination_char:
      print("Goodbye!") 
      sys.exit(0)
    
    # input is not valid
    else:
      print(f"Input: {input_char} is not valid. Please try again.")
      continue






  


  