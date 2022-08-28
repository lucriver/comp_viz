def get_model_choice(models: list) -> str:
  print("Please choose from the following models:")
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
    print(i,val)


def is_valid_input(x: str, valid_x: list) -> bool:
  if x not in valid_x:
    return False
  return True


def handle_model_choice_input(user_input: str, models: list) -> str:
  model = models[int(user_input[0])]
  if "d" in user_input:
    print(describe_model(model))
    input("Press any key to continue...")
    return get_model_choice(models)
  return model


def describe_model(name: str) -> tuple:
  model = {}
  model["yolo3_mobilenet1.0_coco"] = "cool"
  model["center_net_resnet101_v1b_dcnv2_coco"] = "cool2"
  model["yolo3_darknet53_coco"] = "cool3"
  model["faster_rcnn_fpn_syncbn_resnest269_coco"] = "cool4"
  return name, model[name]

def get_selection():
  selections = ["Choose another model"]