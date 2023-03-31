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
        image_name = str(self.map.iloc[index].image)
        image_location = os.path.join(self.root_dir, image_name)
        image = PIL.Image.open(image_location).convert('RGB')
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

        return image, target, image_location

