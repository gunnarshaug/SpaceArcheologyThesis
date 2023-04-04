import os
import PIL
import glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import numpy as np

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
        target["image_location"] = image_name

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

            # If no bounding boxes are found, set them to a tensor of zeros
            if len(target["boxes"]) == 0:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        
        return image, target, image_name

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
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd         
        }
        
        return target

