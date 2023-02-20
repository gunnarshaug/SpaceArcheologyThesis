import torchvision
import utils.not_used
import torchvision
import argparse
import utils.cnf
import utils.general
import os
import torch
import utils.data
import utils.metrics

def evaluate(config, model, device, loader):
  model.eval() 
  true_positives = 0
  false_positives = 0
  false_negatives = 0
  counter = 0
  for images, targets in loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    results = model(images)
    for i, result in enumerate(results):
      nms_prediction = utils.general.apply_nms(results[i], iou_thresh=0.1)
      ground_truth = targets[i]
      iou = torchvision.ops.box_iou(nms_prediction['boxes'].to(device), ground_truth['boxes'].to(device))
      tp, fp, fn = utils.metrics.compute_accuracy(iou)
      true_positives += tp
      false_positives += fp
      false_negatives += fn
      counter += 1
  print("For total of ", counter, " images the results are following:")
  print("True positives: ", true_positives)
  print("False positives: ", false_positives)
  print("False negatives: ", false_negatives)
  print("Recall: ", true_positives / (true_positives + false_negatives))
  print("Precision: ", true_positives / (true_positives + false_positives))
  utils.general.write_results_summary_csv(
    config.model.path,
    counter, 
    true_positives,
    false_positives, 
    false_negatives, 
    config.train.optimizer.lr, 
    config.test.epochs, 
    config.model.name
  )

def parse_args():
  parser = argparse.ArgumentParser(description='Evaluate a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='test dataset',
                      default='datasets/slope/test', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='config/faster_rcnn.yml', type=str)
  parser.add_argument('--model',
                      help='path to model',
                      default='models/slope.pt', type=str)
  return parser.parse_args()


if __name__ == "__main__":
  config = utils.cnf.load("config/faster_rcnn.yml")
  args = parse_args()
  test_loader = utils.data.create_dataloader(config, "test")
  model_path = os.path.join(config.model.path, config.model.name)
  model = torch.load(model_path)
  evaluate(model, test_loader)

