import torch
import os
import argparse
import utils.metrics
import utils.general
import utils.model
import trainers.trainer
import loggers.wandb
import data.dataloaders


def parse_args():
  parser = argparse.ArgumentParser(description='Pytorch Faster R-CNN')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--config',
                      help='location of config file',
                      default='faster_rcnn.yml', type=str)
  return parser.parse_args()

def main():
  args = parse_args()
  cfg_path = os.path.join("config", args.config)

  try:
    config = utils.general.load_cfg(cfg_path)
  except FileNotFoundError as ex:
    print(f"ERROR: Cannot find config file {args.config}")
    print(ex.strerror, ex.filename)
    print(f"stopping program...")
    return
    
  use_cuda = torch.cuda.is_available()

  if not use_cuda:
    print("WARNING: running training with CPU.")
  
  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")
  
  logger = loggers.wandb.WandbLogger(
    config=config
  )

  trainer = trainers.trainer.Trainer(
    logger=logger,
    device=device
  )
  
  data_loaders = data.dataloaders.DataLoaders(**logger.config["data"])

  trainer.train(data_loaders)
  
  trainer.test(data_loaders)

if __name__ == "__main__":
  print(torch.cuda.is_available())
  main()
