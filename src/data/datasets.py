import os
import glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL.Image

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
    """
    def __init__(self,
                 root_dir:str,
                 transform=None) -> None:
        assert root_dir is not None and root_dir != ""
        self.root_dir = root_dir
        self.transform = transform

        path = os.path.join(self.root_dir, "images/*.tif")
        normalized_path = os.path.normpath(path)
        self.images = sorted(glob.glob(normalized_path))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> tuple:
        image_path = self.images[index]
        image_name = os.path.basename(image_path)
        image = PIL.Image.open(image_path).convert('RGB')

        target = self.get_annotation(index, image_name)

        if self.transform is not None:
            image_numpy = np.array(image)
            
            transformed = self.transform(
                image=image_numpy,
                bboxes=target["boxes"],
                class_labels=target["labels"]
            )
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target["boxes"] = boxes

            if len(target["boxes"]) == 0:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        
        return image, target

    def get_annotation(self, index, image_name):
        annotation_path = os.path.join(self.root_dir, "labels", image_name.replace(".tif", ".xml"))
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        area = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # There is only one class in this example
            area.append((xmax - xmin) * (ymax - ymin))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(labels), dtype=torch.int64)
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        
        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd         
        }


class Dataset2022(Dataset):
    """
        Assumes the following root directory structure:
        - root
            - data
            - classification.csv
    """
    def __init__(self, root_dir, transform=None) -> None:
        self.img_dir = os.path.join(root_dir, "data")
        # self.img_dir = os.path.normpath(os.path.join(root_dir, "data"))
        self.images = [image for image in sorted(os.listdir(self.img_dir)) if image.split(".")[1].lower() == "png"]
        
        # path = os.path.join(root_dir, "data/*.png")
        # normalized_path = os.path.normpath(path)
        # self.images = sorted(glob.glob(normalized_path))
        print("root_dir: ", root_dir)
        print("self.images: ", self.images)
        # print("normalized_path: ", normalized_path)
        # print("path: ", path)
        # print("listdir path:", os.listdir(path))
        print("listdir root_dir:", os.listdir(root_dir))

        annotations_file = os.path.join(root_dir, "classification.csv")
        self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        print("IMG PATH: ", img_path)
        img_name = os.path.basename(img_path)
        print("IMG NAME: ", img_name)

        image = PIL.Image.open(img_path).convert('RGB')

        records = self.img_labels.loc[self.img_labels.filename == img_name]
        bounding_boxes, labels, area = []

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

        tensor_bounding_boxes = torch.as_tensor(transformed_bounding_boxes, dtype=torch.float32)
        iscrowd = torch.zeros((tensor_bounding_boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = tensor_bounding_boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target