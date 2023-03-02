import csv
import os
import torch
import datetime
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_pretrained_frcnn(num_classes=2):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model


def train_one_epoch(model, device, loader, optimizer, loss_hist):
  model.train()
  for batch_idx, (inputs, labels) in enumerate(loader):
    images = list(image.to(device) for image in inputs)
    targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
    output = model(images, targets)

    losses = sum(loss for loss in output.values())
    loss_value = losses.item()

    loss_hist.update(loss_value)

    # compute loss and its gradients
    optimizer.zero_grad()
    losses.backward()
    # adjust learning weights based on the gradients we just computed
    optimizer.step()


def validate(model, device, val_loader, stats):
  model.eval() 
  with torch.no_grad():
    for images, targets in val_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      output = model(images)
      for i, result in enumerate(output):
        nms_prediction = _apply_nms(result, iou_thresh=0.1)
        ground_truth = targets[i]
        iou = torchvision.ops.box_iou(
          nms_prediction['boxes'].to(device),
          ground_truth['boxes'].to(device)
        )
        predicted_boxes_count, gt_boxes_count = list(iou.size())

        if predicted_boxes_count == 0 and gt_boxes_count == 0:
          continue
        
        tp, fp, fn = _compute_accuracy(iou)
        stats.update(tp, fp, fn)
  
def test(config, model, device, loader, stats, tb_writer):
  model.eval() 

  for idx, (images, targets ) in enumerate(loader):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(images)
    for i, result in enumerate(output):
      nms_prediction = _apply_nms(result, iou_thresh=0.1)
      ground_truth = targets[i]
      iou = torchvision.ops.box_iou(
        nms_prediction['boxes'].to(device), 
        ground_truth['boxes'].to(device)
      )

      tp, fp, fn = _compute_accuracy(iou)
      stats.update(tp, fp, fn)

      if(idx == 0 and i == 0):
        tb_writer.add_image_with_boxes("Fist Image Prediction", images[0], nms_prediction['boxes'])
        tb_writer.add_image_with_boxes("Fist Image Ground Truth", images[0], ground_truth['boxes'])

  print("For total of ", stats.counter, " images the results are following:")
  print("True positives: ", stats.get_true_positives())
  print("False positives: ", stats.get_false_positives())
  print("False negatives: ", stats.get_false_negatives())
  print("Recall: ", stats.get_recall())
  print("Precision: ", stats.get_precision())
  _write_results_summary_csv(stats, config)


# the function takes the original prediction and the iou threshold.
def _apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction

def _write_results_summary_csv(stats, cfg):
  timestamp = datetime.datetime.now().strftime('%d.%m.%Y_%H.%M.%S')
  model_name = "{}_{}.csv".format(cfg.model.name, timestamp)
  with open(os.path.join(cfg.model.path, model_name), "w", newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([f"{cfg.name}"])
    csv_writer.writerow([f"# of Train Images: {cfg.train.images}"])
    csv_writer.writerow([f"# of Train Images: {cfg.train.val_images}"])
    csv_writer.writerow([f"# of Test Images: {cfg.test.images}"])
    csv_writer.writerow([f"{cfg.name}"])
    csv_writer.writerow([f"# of Test Images: {stats.get_counter()} "])
    csv_writer.writerow([f"# of True Positives: {stats.get_true_positives()}"])
    csv_writer.writerow([f"# of False Positives: {stats.get_false_positives()}"])
    csv_writer.writerow([f"# of False Negatives: {stats.get_false_negatives()}"])
    csv_writer.writerow([f"Precision: {stats.get_precision()}"])
    csv_writer.writerow([f"Recall: {stats.get_recall()}"])
    csv_writer.writerow("")
    csv_writer.writerow(["Model Settings:"])
    csv_writer.writerow([f"# of Train Epochs: {cfg.train.epochs}"])
    csv_writer.writerow([f"# of Test Epochs: {cfg.test.epochs}"])
    csv_writer.writerow([f"Optimizer: {cfg.train.optimizer.name}"])
    csv_writer.writerow([f"Learning Rate: {cfg.train.optimizer.lr}"])
    csv_writer.writerow([f"Momentum: {cfg.train.optimizer.momentum}"])
    csv_writer.writerow([f"Weight Decay: {cfg.train.optimizer.weight_decay}"])

def _compute_accuracy(iou):
  predicted_boxes_count, gt_boxes_count = list(iou.size())
    
  fp = 0
  tp = 0

  for box in iou:
    valid_hits = [i for i, x in enumerate(box) if x > 0.5 ]
    if len(valid_hits) == 0:
      fp = fp + 1
      continue
    tp = tp + 1
    

  fn = gt_boxes_count - tp
  return tp, fp, fn
