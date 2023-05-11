import os
import cv2
import csv
import torch
import numpy as np
import albumentations as a
import albumentations.pytorch.transforms
import utils.general
import demo.calculate_results

"""
Scan of large Slope image of test area. Set overlap to 0.5 to use overlap scan with 50% overlap.
"""

class Demo:
  def __init__(self, 
               model,
               device: str,
               store_dir: str,
               overlap: float=0.0,
               img_split_dim: dict={"width":400, "height": 400}
              ) -> None:
    
    self.model = model
    self.device = device
    self.store_dir = store_dir
    self.img_split_dim = img_split_dim
    self.overlap = overlap

  def _start_points(self, size, split_size):
    points = [0]
    stride = int(split_size * (1 - self.overlap))
    counter = 1
    while True:
      pt = stride * counter
      if pt + split_size >= size:
        points.append(size - split_size)
        break
      else:
        points.append(pt)
        counter += 1
    return points
  def analyze_image(self, image_path: str, result_csv_name:str, utm_east=None, utm_north=None, pixel_size=None) -> None:
    self._cleanup()

    image = cv2.imread(image_path)
    images = self._split_image(image)

    result = self._predict(images)

    normalized_bboxes = self._normalize_coordinates(result)
    self._save_image(image, normalized_bboxes)
    self._generate_result_csv(result_csv_name, normalized_bboxes)

    self.calculateCoordinates(utm_east, utm_north, pixel_size)

  @property
  def transform(self):
    return a.Compose([
        a.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        albumentations.pytorch.transforms.ToTensorV2()
      ])
  
  def _split_image(self, image):
    img_height, img_width, _ = image.shape
    X_points = self._start_points(img_width, self.img_split_dim["width"])
    Y_points = self._start_points(img_height, self.img_split_dim["height"])
    
    images = np.empty(shape=(len(X_points), len(Y_points)))
    images = [[[] for i in range(len(X_points))] for i in range(len(Y_points))]
    count = 0

    for i, y_point in enumerate(Y_points):
      for j, x_point in enumerate(X_points):
        split = image[y_point:y_point +  self.img_split_dim["height"],
                      x_point:x_point +  self.img_split_dim["width"]]
        
        images[i][j] = split
        
        #storing
        image_dir = os.path.join(
          self.store_dir,
          "parts",
        )
        utils.general.ensure_existing_dir(image_dir)
          
        image_name = 'splitted_{}_{}.png'.format(i, j)
        
        image_path = os.path.join(image_dir, image_name)

        cv2.imwrite(image_path, split)
        count += 1
    return np.array(images)

  def _predict(self, images):
    image_dir = os.path.join(self.store_dir, "predictions")
    utils.general.ensure_existing_dir(image_dir)
    self.model.eval()
    bboxes = [[[] for i in range(images.shape[0])] for i in range(images.shape[1])]
    low_scores_count = 0
    
    for j, row in enumerate(images):
      for i, image in enumerate(row):
        image_numpy = np.array(image)
        transformed_image = self.transform(image=image_numpy)['image']
        with torch.no_grad():
          prediction = self.model([transformed_image.to(self.device)])[0]
          nms_prediction = utils.general.apply_nms(prediction, iou_thresh=0.1)
          bboxes[i][j] = nms_prediction['boxes']
          for box in nms_prediction['boxes']:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

          image_name = '{}_{}.png'.format(i, j)
          image_path = os.path.join(image_dir, image_name)
          cv2.imwrite(image_path, image)
    return bboxes

  def _normalize_coordinates(self, bboxes):
    normalized_boxes = []
    for x, row in enumerate(bboxes):
      for y, boxes in enumerate(row):
        x_overlap_padding = self.overlap * x * self.img_split_dim["width"]
        y_overlap_padding = self.overlap * y * self.img_split_dim["height"]
        x_padding, y_padding = x * self.img_split_dim["width"], y * self.img_split_dim["height"]
        for i, box in enumerate(boxes):
          x_min, y_min, x_max, y_max = box
          normalized_boxes.append([
            x_min + x_padding - x_overlap_padding,
            y_min + y_padding - y_overlap_padding,
            x_max + x_padding - x_overlap_padding,
            y_max + y_padding - y_overlap_padding,
            "{}_{}".format(x, y)                              
          ])
    return normalized_boxes

  def _save_image(self, image, bboxes):
    """
    Saves PNG image with bounding boxes.
    """
    for box in bboxes:
      x_min, y_min, x_max, y_max, label = box
      cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    image_path = os.path.join(self.store_dir, 'Composite_Slope_Results_without_Overlap_400.png') 
    cv2.imwrite(image_path, image)


  def _generate_result_csv(self, csv_name, bboxes):  
    headers = ['xmin', 'ymin', 'xmax', 'ymax']
    result_csv_path = os.path.join(self.store_dir, f"{csv_name}.csv")

    with open(result_csv_path, 'w', encoding='UTF8', newline="") as f:
      writer = csv.writer(f)
      writer.writerow(headers)
      for _, box in enumerate(bboxes):
        xmin, ymin, xmax, ymax, _ = box
        writer.writerow([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
    self.result_csv_path = result_csv_path

  def calculateCoordinates(self, utm_left:float, utm_top:float, pixel_size:float=0.25):
    demo.calculate_results.calculateCoordinates(self.result_csv_path, utm_left, utm_top, pixel_size)

  def _cleanup(self) -> None:
    dir = os.path.join(self.store_dir, "parts")
    for f in os.listdir(dir):
      os.remove(os.path.join(dir, f))
    
    dir = os.path.join(self.store_dir, "predictions")
    for f in os.listdir(dir):
      os.remove(os.path.join(dir, f))
    