import csv
import os
import torchvision
import torch
import utils.metrics
import utils.data
import utils.general
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_pretrained_frcnn(num_classes):
  # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
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
        
        tp, fp, fn = utils.metrics.compute_accuracy(iou)
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

      tp, fp, fn = utils.metrics.compute_accuracy(iou)
      stats.update(tp, fp, fn)

      if(idx == 0 and i == 0):
        # tb_writer.add_image_with_boxes("Fist Image", images[0], targets[0])
        tb_writer.add_image_with_boxes("Fist Image", images[0], iou)

  print("For total of ", stats.counter, " images the results are following:")
  print("True positives: ", stats.get_true_positives())
  print("False positives: ", stats.get_false_positives())
  print("False negatives: ", stats.get_false_negatives())
  print("Recall: ", stats.get_recall())
  print("Precision: ", stats.get_precision())
  _write_results_summary_csv(
    config.model.path,
    stats.counter, 
    stats.get_true_positives(),
    stats.get_false_positives(), 
    stats.get_false_negatives(), 
    config.train.optimizer.lr, 
    config.test.epochs, 
    config.model.name
  )


# the function takes the original prediction and the iou threshold.
def _apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction

def _write_results_summary_csv(path, num_test_images, true_positives, false_positives, false_negatives, learning_rate, num_epoch, result_filename):
  with open(os.path.join(path, result_filename + ".csv"), "w", newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([f"# of Test Images: {num_test_images} "])
    csv_writer.writerow([f"# of True Positives: {true_positives}"])
    csv_writer.writerow([f"# of False Positives: {false_positives}"])
    csv_writer.writerow([f"# of False Negatives: {false_negatives}"])
    csv_writer.writerow([f"Precision: {true_positives/(true_positives+false_positives)}"])
    csv_writer.writerow([f"Recall: {true_positives/(true_positives+false_negatives)}"])
    csv_writer.writerow("")
    csv_writer.writerow(["Model Settings:"])
    csv_writer.writerow([f"# of Epochs: {num_epoch}"])
    csv_writer.writerow([f"Learning Rate: {learning_rate}"])