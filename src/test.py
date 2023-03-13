import os
import argparse
import utils.general
import torch
from data.dataloaders import(MainDataLoader)
import utils.metrics
import utils.model
import torchvision
from torch.utils.tensorboard import SummaryWriter

def parse_args():
  parser = argparse.ArgumentParser(description='Predict burial mounds')
  parser.add_argument('--dataset', dest='dataset',
                      help='test dataset',
                      default='datasets/slope/test', type=str)
  parser.add_argument('--config', 
                      help='config file path',
                      default='faster_rcnn.yml', type=str)
  parser.add_argument('--model',
                      help='path to model',
                      default='models/slope.pt', type=str)
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

  test_loader = utils.data.MoundDataset2022.get_dataloader(config, "test")
  model_path = os.path.join(config["model"]["path"], config["model"]["name"])
  use_cuda = torch.cuda.is_available()

  device = torch.device("cuda" if use_cuda else "cpu")

  model = torch.load(model_path, device)

  loader_kwargs = {
    "data_dir": config["dataset"]["path"],
    "dataset_type": config["dataset"]["type"],
    "transform_opts": config["transform"],
    "num_workers": ["dataloader"]["num_workers"],
    "batch_size": config["dataloader"]["batch_size"],
  }

  test_loader = MainDataLoader(
    mode="test",
    **loader_kwargs
  )

  test_stats = utils.metrics.Metrics()
  log_dir = 'tensorboard/frcnn_tester_{}'.format(config["timestamp"])
  tb_writer = SummaryWriter(log_dir)

  model.eval() 

  for idx, (images, targets ) in enumerate(test_loader):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(images)
    for i, result in enumerate(output):
      nms_prediction = utils.model.apply_nms(result, iou_thresh=0.1)
      ground_truth = targets[i]
      iou = torchvision.ops.box_iou(
        nms_prediction['boxes'].to(device), 
        ground_truth['boxes'].to(device)
      )
      #OBS! update to correct function!
      tp, fp, fn = utils.model.compu(iou)
      test_stats.update(iou)

      if(idx == 0 and i == 0):
        tb_writer.add_image_with_boxes("Fist Image Prediction", images[0], nms_prediction['boxes'])
        tb_writer.add_image_with_boxes("Fist Image Ground Truth", images[0], ground_truth['boxes'])

  print("For total of ", test_stats.counter, " images the results are following:")
  print("True positives: ", test_stats.get_true_positives())
  print("False positives: ", test_stats.get_false_positives())
  print("False negatives: ", test_stats.get_false_negatives())
  print("Recall: ", test_stats.get_recall())
  print("Precision: ", test_stats.get_precision())
  utils.model.write_results_summary_csv(test_stats, config)

  #TODO: predict and visualize results.

if __name__ == "__main__":
  main()
