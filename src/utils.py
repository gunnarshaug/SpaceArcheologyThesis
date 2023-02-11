import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import torch
import csv
import os
# from config import root

def collate_fn(batch):
    """
    copy from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    """
    return tuple(zip(*batch))


def display_image(image, bboxes, gt_boxes=None):
    plt.imshow(image[0])
    ax = plt.gca()
    plt.grid(False)
    plt.axis('off')               

    for box in bboxes["boxes"]:
      x_min, y_min, x_max, y_max = box
      width = x_max - x_min
      height = y_max - y_min
      rect = patches.Rectangle((x_min, y_min), width, height, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
    if gt_boxes != None:
      for box in gt_boxes["boxes"]:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()
      
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class Stats:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.counter = 0

    def send(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.counter += 1

    def get_precision(self):
      if (self.tp + self.fp) == 0:
        return 0
      else:
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
      if (self.tp + self.fn) == 0:
        return 0
      else: 
        return self.tp / (self.tp + self.fn)

    def get_true_positives(self):
      return self.tp

    def get_false_positives(self):
      return self.fp

    def get_false_negatives(self):
      return self.fn

    def get_counter(self):
      return self.counter

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
  return torchvision.transforms.ToPILImage()(img).convert('RGB')


def compute_accuracy(iou):
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

def save_model(model, path):
  torch.save(model, path)

def load_model(path):
  model = torch.load(path)
  return model


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