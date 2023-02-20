import torchvision
import csv
import os

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction


def write_results_summary_csv(path, num_test_images, true_positives, false_positives, false_negatives, learning_rate, num_epoch, result_filename):
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