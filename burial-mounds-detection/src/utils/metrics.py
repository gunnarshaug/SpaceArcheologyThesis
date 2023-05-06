import torch

class Metrics:
  def __init__(self):
    self.tp = 0
    self.fp = 0
    self.fn = 0
    self.counter = 0
    self.is_improving = False

  def update(self, iou: torch.Tensor):
    precision_prev = self.precision
    
    tp, fp, fn = self._compute_metrics(iou)
    self.tp += tp
    self.fp += fp
    self.fn += fn
    self.counter += 1
    
    self.is_improving = self.precision >= precision_prev

  @property
  def precision(self) -> float:
    if (self.tp + self.fp) == 0:
      return 0
    else:
      return self.tp / (self.tp + self.fp)

  @property
  def recall(self) -> float:
    if (self.tp + self.fn) == 0:
      return 0
    else: 
      return self.tp / (self.tp + self.fn)
    
  @property
  def f1_score(self) -> float:
    if (self.precision+self.recall) == 0:
      return 0
    else: 
      return (2*self.precision*self.recall) / (self.precision + self.recall)
    
  @property
  def true_positives(self)-> int:
    return self.tp

  @property
  def false_positives(self)-> int:
    return self.fp

  @property
  def false_negatives(self)-> int:
    return self.fn
  
  def _compute_metrics(self, iou : torch.Tensor) -> tuple:
    predicted_boxes_count, gt_boxes_count = list(iou.size())
      
    fp = 0
    tp = 0

    for box in iou:
      valid_hits = [i for i, x in enumerate(box) if x > 0.5 ]
      if len(valid_hits) == 0:
        fp += 1
        continue
      tp += 1
      
    fn = gt_boxes_count - tp
    return tp, fp, fn

class Averager:
  def __init__(self):
    self.total = 0.0
    self.iterations = 0.0

  def update(self, value):
    self.total += value
    self.iterations += 1

  @property
  def value(self):
    if self.iterations == 0:
        return 0
    else:
        return 1.0 * self.total / self.iterations

  def reset(self):
    self.total = 0.0
    self.iterations = 0.0
