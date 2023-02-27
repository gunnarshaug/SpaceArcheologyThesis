import os
import argparse
import utils.general
import torch
import utils.data
import utils.metrics
import utils.model

def parse_args():
  parser = argparse.ArgumentParser(description='Predict burial mounds')
  parser.add_argument('--dataset', dest='dataset',
                      help='test dataset',
                      default='datasets/slope/test', type=str)
  parser.add_argument('--config', 
                      help='config file path',
                      default='config/faster_rcnn.yml', type=str)
  parser.add_argument('--model',
                      help='path to model',
                      default='models/slope.pt', type=str)
  return parser.parse_args()


def main():
  args = parse_args()
  config = utils.general.load_cfg(args.config)
  test_loader = utils.data.get_dataloader(config, "test")
  model_path = os.path.join(config.model.path, config.model.name)
  use_cuda = torch.cuda.is_available()

  device = torch.device("cuda" if use_cuda else "cpu")

  model = torch.load(model_path, device)
  #TODO: predict and visualize results.

# if __name__ == "__main__":
  # main()
