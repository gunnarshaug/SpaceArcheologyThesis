import torch
import pandas as pd
class Stats:
  def __init__(self):
    self.tp = 0
    self.fp = 0
    self.fn = 0
    self.counter = 0

  def update(self, iou: torch.Tensor):
    tp, fp, fn = self._compute_accuracy(iou)
    self.tp += tp
    self.fp += fp
    self.fn += fn
    self.counter += 1

  def get_precision(self) -> float:
    if (self.tp + self.fp) == 0:
      return 0
    else:
      return self.tp / (self.tp + self.fp)

  def get_recall(self) -> float:
    if (self.tp + self.fn) == 0:
      return 0
    else: 
      return self.tp / (self.tp + self.fn)

  def get_true_positives(self)-> int:
    return self.tp

  def get_false_positives(self)-> int:
    return self.fp

  def get_false_negatives(self)-> int:
    return self.fn

  def get_counter(self) -> int:
    return self.counter
  
  def _compute_accuracy(iou : torch.Tensor) -> tuple:
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

class Averager:
  def __init__(self):
    self.current_total = 0.0
    self.iterations = 0.0

  def update(self, value):
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


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        pd.DataFrame()
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)