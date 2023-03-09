import os
import PIL
import xml
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):
    """
    Assumes a map.txt file which contains two columns: the relative path to the images and the corresponding annotation xml file. 

    Example folder of PascalVOC annotation created by ArcGIS Pro:
    - root
        - images
            - 0000.tif
            - 0001.tif
            - ...
        - labels
            - 0000.xml
            - 0001.xml
            - ...
        - map.txt
        - esri_accumulated_stats.json (not relevant)
        - esri_model_definition.emd (not relevant)
        - stats.txt (not relevant)
    """
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        map_path = os.path.join(root_dir, "map.txt")
        self.map = pd.read_csv(map_path, sep=' ', header=None, names=["image", "annotation"])

    def __len__(self) -> int:
        return len(self.map)

    def __getitem__(self, index) -> tuple:
        image_file = os.path.join(self.root_dir, str(self.map.iloc[index].image))
        image = PIL.Image.open(image_file).convert('RGB')
        bounding_boxes = []
        area = []
        labels = []
        no_objects = 0
        
        has_bounding_boxes = str(self.map.iloc[index].annotation) != "nan"
        if has_bounding_boxes:

            annotation_file = os.path.join(self.root_dir, str(self.map.iloc[index].annotation))

            annotation = xml.etree.ElementTree.parse(annotation_file).getroot()
            no_objects = len(annotation.findall("object"))
            for object in annotation.findall("object"):
                labels.append("mound")
                bbox = object.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                bounding_boxes.append([xmin, ymin, xmax, ymax])
                area.append((xmax - xmin) * (ymax - ymin))

        area = torch.as_tensor(area, dtype=torch.float32)

        if self.transform is not None:
            image_numpy = np.array(image)

            transformed = self.transform(
                image=image_numpy,
                bboxes=bounding_boxes,
                class_labels=labels
            )
            image = transformed['image']
            bounding_boxes = transformed['bboxes']
            
            if len(bounding_boxes) == 0:
                bounding_boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        labels = torch.ones(no_objects, dtype=torch.int64) # there is only one class

        tensor_bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)
        iscrowd = torch.zeros((tensor_bounding_boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = tensor_bounding_boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target


class Dataset2022(Dataset):
    """
        Assumes the following root directory structure:
        - root
            - data
            - classification.csv
    """
    def __init__(self, root_dir, transform=None) -> None:
        self.img_dir = os.path.join(root_dir, "data")
        self.images = [image for image in sorted(os.listdir(self.img_dir))]

        annotations_file = os.path.join(root_dir, "classification.csv")
        self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform

    def __len__(self) -> int:
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
        
        labels = torch.ones(len(records), dtype=torch.int64) # there is only one class
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
