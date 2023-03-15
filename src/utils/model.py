import csv
import os
import torchvision


def apply_nms(orig_prediction, iou_thresh: float = 0.3) -> dict:
  """
  :param iou_thresh: iou threshold
  :param orig_prediction: original prediction
  :returns the final iou threshold
  """
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction

def write_results_summary_csv(metric_tracker, cfg) -> None:
  model_name = "{}_{}.csv".format(cfg["model"]["name"], cfg["timestamp"])
  with open(os.path.join(cfg["model"]["path"], model_name), "w", newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    # csv_writer.writerow([f"{cfg["name"]}"])
    # csv_writer.writerow([f"# of Train Images: {cfg["train"]["images"]}"])
    # csv_writer.writerow([f"# of Train Images: {cfg["train"]["val_images"]}"])
    # csv_writer.writerow([f"# of Test Images: {cfg["test"]["images"]}"])
    csv_writer.writerow([f"{cfg['name']}"])
    csv_writer.writerow([f"# of Test Images: {metric_tracker.counter} "])
    csv_writer.writerow([f"# of True Positives: {metric_tracker.true_positives}"])
    csv_writer.writerow([f"# of False Positives: {metric_tracker.false_positives}"])
    csv_writer.writerow([f"# of False Negatives: {metric_tracker.false_negatives}"])
    csv_writer.writerow([f"Precision: {metric_tracker.precision}"])
    csv_writer.writerow([f"Recall: {metric_tracker.recall}"])
    csv_writer.writerow("")
    csv_writer.writerow(["Model Settings:"])
    csv_writer.writerow([f"# of Train Epochs: {cfg['train']['epochs']}"])
    csv_writer.writerow([f"# of Test Epochs: {cfg['test']['epochs']}"])
    csv_writer.writerow([f"Optimizer: {cfg['train']['optimizer']['name']}"])
    csv_writer.writerow([f"Learning Rate: {cfg['train']['optimizer']['lr']}"])
    csv_writer.writerow([f"Momentum: {cfg['train']['optimizer']['momentum']}"])
    csv_writer.writerow([f"Weight Decay: {cfg['train']['optimizer']['weight_decay']}"])

