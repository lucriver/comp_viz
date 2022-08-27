from comp_viz.comp_viz import CompViz, CompVizModel

def handle_input(input):
  if "d" in input:
    describe_

if __name__ == "__main__":
  print("Welcome to Lucas' computer vision detector!")

  print("Please choose a computer vision model:")
  for i, network in enumerate(CompViz.get_networks()):
    print(i,network)
  print("(For details about a particular model, enter its corresponding numerical value followed by \'d'. Ex: 0d)")