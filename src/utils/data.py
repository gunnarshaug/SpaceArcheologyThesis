import os
import cv2
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import albumentations as a
import albumentations.pytorch.transforms


class Mound(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.bounding_boxes = pd.read_csv(annotations_file) #duplicate?

        self.img_dir = img_dir
        self.images = [image for image in sorted(os.listdir(img_dir))]
        self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform

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
        labels.append('mounds')
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
      #NOTE: this will break if self.transform is false. transformed_bounding_boxes not defined
      tensor_bounding_boxes = torch.as_tensor(transformed_bounding_boxes, dtype=torch.float32)
      iscrowd = torch.zeros((tensor_bounding_boxes.shape[0],), dtype=torch.int64)

      target = {}
      target["boxes"] = tensor_bounding_boxes
      target["labels"] = labels
      target["image_id"] = torch.tensor([index])
      target["area"] = area
      target["iscrowd"] = iscrowd

      return image, target
      

def get_dataloader(config, mode="train"):
    path = config.dataset.path
    if mode == "train":
        data_path = path.train
    elif mode == "val":
        data_path = path.val
    elif mode == "test":
        data_path = path.test
    else:
       return
    
    isTrain = mode=="train"
    transform = _get_transform(config.dataset.transform, isTrain)

    dataset = Mound(
        os.path.join(path.base, data_path.labels),
        os.path.join(path.base, data_path.images),
        transform
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=config.dataset.loader.num_workers,
        collate_fn=_collate_fn
    )

def _get_transform(dimensions, train=False):
    transforms = []
    transforms.append(a.Resize(dimensions.width, dimensions.height))

    if train:
        transforms.append(a.HorizontalFlip(p=0.5))
        transforms.append(a.RandomBrightnessContrast(p=0.2))

    transforms.append(
        a.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    transforms.append(albumentations.pytorch.transforms.ToTensorV2())

    return a.Compose(
        transforms,
        bbox_params=a.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )

def _collate_fn(batch):
    """
    copy from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    """
    return tuple(zip(*batch))


# if __name__ == "__main__":
#     import general
#     import matplotlib.pyplot as plt
#     cfg = general.load_cfg("config/faster_rcnn.yml")
#     dataloader = create_dataloader(cfg)
#     features, labels = next(iter(dataloader))

#     print()
#     img = features[0].squeeze()
#     img = (img.T).detach().numpy()
#     plt.imshow(img)
#     plt.show()
    
