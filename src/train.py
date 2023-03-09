import torch.utils.tensorboard
import torch
import os
import argparse
import utils.metrics
import utils.general
import utils.model
from data.dataloaders import(MainDataLoader)
from trainer.trainer import (Trainer)


def parse_args():
  parser = argparse.ArgumentParser(description='Pytorch Faster R-CNN')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training. Not recommended.')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--save-model', action='store_true', default=True,
                      help='For Saving the current Model')
  parser.add_argument('--config',
                      help='location of config file',
                      default='config/faster_rcnn.yml', type=str)
  return parser.parse_args()

def main():
  args = parse_args()

  try:
    config = utils.general.load_cfg(args.config)
  except FileNotFoundError as ex:
    print(f"ERROR: Cannot find config file {args.config}")
    print(ex.strerror, ex.filename)
    print(f"stopping program...")
    return
    
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  if not use_cuda:
    print("WARNING: running training with CPU.")
  
  # this is to get the same result in every pass
  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  data_config = config["dataset"]
  loader_kwargs = {
    "data_dir": data_config["path"],
    "dataset_type": data_config["type"],
    "transform_opts": data_config["transform"],
    "num_workers": data_config["loader"]["num_workers"],
    "batch_size": data_config["loader"]["batch_size"],
  }

  train_loader = MainDataLoader(
    mode="train",
    **loader_kwargs
  )

  val_loader = MainDataLoader(
    mode="val",
    **loader_kwargs
  )

  model = utils.model.get_pretrained_frcnn()
  model.to(device)

  optimizer_config = config["trainer"]["optimizer"]
  optimizer = torch.optim.SGD(
      params=[p for p in model.parameters() if p.requires_grad],
      lr=optimizer_config["lr"],
      momentum=optimizer_config["momentum"],
      weight_decay=optimizer_config["weight_decay"]
  )

  scheduler_config = config["trainer"]["scheduler"]
  lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size= scheduler_config["step_size"],
    gamma=scheduler_config["gamma"]
  )

  trainer = Trainer(
    model=model, 
    optimizer=optimizer,
    config=config,
    device=device, 
    dataloader=train_loader,
    val_dataloader=val_loader, 
    lr_scheduler=lr_scheduler,
    save_model=args.save_model
  )

  trainer.train()

if __name__ == "__main__":
  print(torch.cuda.is_available())
  main()
