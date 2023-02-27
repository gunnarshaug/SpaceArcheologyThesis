class Stats:
  def __init__(self):
      self.tp = 0
      self.fp = 0
      self.fn = 0
      self.counter = 0

  def update(self, tp, fp, fn):
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

