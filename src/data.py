import numpy as np
import pandas as pd
import cv2
import torch
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class Mound(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.bounding_boxes  = pd.read_csv(annotations_file)

        self.img_dir = img_dir
        self.images = [image for image in sorted(os.listdir(img_dir))]
        self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform

    # def generate_original_image_data(self):
    #   image_data = []
    #   for image in self.images:
    #     im = Image.open(os.path.join(test_data, image))
    #     width, height = im.size
    #     image_data.append((image, width, height))
    #   return image_data

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
      img_name = self.images[index]
      img_path = os.path.join(self.img_dir, img_name)
      image = cv2.imread(img_path)

      records = self.img_labels.loc[self.img_labels.filename == img_name]
      bounding_boxes = []
      labels = []
      area = []

      for i in range(len(records)):
        record = records.iloc[i]
        labels.append('mound')
        bounding_boxes.append([
          record.xmin,
          record.ymin,
          record.xmax,
          record.ymax,
        ])
        area.append((record.xmax - record.xmin) * (record.ymax - record.ymin))

      area = torch.as_tensor(area, dtype=torch.float32)

      
      if self.transform is not None:
        image_numpy = np.array(image)

        transformed = self.transform(
          image=image_numpy,
          bboxes=bounding_boxes,
          class_labels=labels
        )
        image = transformed['image']
        transformed_bounding_boxes = transformed['bboxes']
        
        if len(bounding_boxes) == 0:
          transformed_bounding_boxes = torch.zeros((0, 4), dtype=torch.float32)

      labels = torch.ones(len(records), dtype=torch.int64)
      
      tensor_bounding_boxes = torch.as_tensor(transformed_bounding_boxes, dtype=torch.float32)
      iscrowd = torch.zeros((tensor_bounding_boxes.shape[0],), dtype=torch.int64)

      target = {}
      target["boxes"] = tensor_bounding_boxes
      target["labels"] = labels
      target["image_id"] = torch.tensor([index])
      target["area"] = area
      target["iscrowd"] = iscrowd

      return image, target