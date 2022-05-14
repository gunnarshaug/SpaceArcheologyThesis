import os
import sys
from datetime import datetime
from pathlib import Path

import albumentations.pytorch
import csv
import cv2
import numpy as np
import torch as torch
import albumentations as A
import torchvision

from app.repositories import models_repository
from app.repositories.models_repository import get_by_id
from app.repositories.results_repository import store_results
from app.services.csv import store_results_to_csv
from app.services.images import split_image, save_image_with_bboxes
from app.services.models.setup.post.Archaeological_Mounds_Calculate_Coordinates import calculateCoordinatesForSAR
from app.services.models.setup.post.Burial_Mounds_Calculate_Coordinates import calculateCoordinatesForBurialMounds
from app.services.models.utils.Compare_Burial_Mound_Results import compare_burial_mound_results
from app.services.models.utils.Exclude_Burial_Mound_Results import exclude_burial_mounds_by_filters

MODELS_PATH = os.environ.get('MODELS_PATH')
ANALYZED_PATH = os.environ.get('ANALYZED_PATH')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_height = 448
image_width = 448


post_setup_procedure_by_file = {
    "model_120_no_optim_SAR_best": calculateCoordinatesForSAR,
    "model_120_no_optim_slope_best": calculateCoordinatesForBurialMounds,
    "model_200_no_optim_msrm_x1_best": calculateCoordinatesForBurialMounds,
    "model_200_no_optim_msrm_x2_best": calculateCoordinatesForBurialMounds,
    "model_200_no_optim_slrm_best": calculateCoordinatesForBurialMounds
}


def create_folder(title):
    now = datetime.now()
    date_time = now.strftime("%m%d%Y-%H%M%S")
    folder_name = "{}_{}".format(title, date_time)
    path = os.path.join(ANALYZED_PATH, folder_name)
    os.mkdir(path)
    return path


def analyze_image(image, model_id, title, overlap, cut_size, custom_params):
    model_data = get_by_id(model_id)
    model_path = model_data['path']
    model = load_model(model_path)
    images = split_image(image, overlap, cut_size, False)
    bboxes = get_bounding_boxes(images, model)
    normalized_bboxes = normalize_coordinates(bboxes, split_size=cut_size)
    solution_folder = create_folder(title)

    csv_path = store_results_to_csv(normalized_bboxes, solution_folder)
    image_path = save_image_with_bboxes(image, normalized_bboxes, solution_folder)
    store_results(model_id, Path(solution_folder).name)

    model_path_stem = Path(model_path).stem
    if model_path_stem in post_setup_procedure_by_file.keys():
        cb = post_setup_procedure_by_file.get(model_path_stem)
        cb({
            'normalized_boxes': normalized_bboxes,
            'csv_path': csv_path,
            'image_path': image_path,
            'custom_params': custom_params
        })
    return solution_folder

def filter_results(data):
    target_folder = create_folder("_filtering")
    filtered_results = []
    i = 0
    for entry in data:
        target_path = entry.get('source')
        min_size = entry.get('smaller_than') if entry.get('target_path') is not None else 0
        max_size = entry.get('bigger_than') if entry.get('target_path') is not None else sys.maxsize
        ratio = entry.get('ratio') if entry.get('target_path') is not None else sys.maxsize
        res = exclude_burial_mounds_by_filters(get_results_file(target_path), target_folder, min_size, max_size, ratio, i)
        filtered_results.append(res)
        i += 1
    if len(filtered_results) > 1:
        src1 = filtered_results[0]
        src2 = filtered_results[1]
        src3 = filtered_results[2] if len(filtered_results) > 2 else None
        src4 = filtered_results[3] if len(filtered_results) > 3 else None
        compare_burial_mound_results(target_folder, src1, src2, src3, src4)
    return target_folder



base_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    albumentations.pytorch.ToTensorV2()
])


def normalize_coordinates(bboxes, overlap=0, split_size=0):
    normalized_boxes = []
    for x, row in enumerate(bboxes):
        for y, boxes in enumerate(row):
            x_overlap_padding = overlap * x * split_size
            y_overlap_padding = overlap * y * split_size
            x_padding, y_padding = x * split_size, y * split_size
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



def get_bounding_boxes(images, model):
    model.eval()
    bboxes = [[[] for i in range(images.shape[0])] for i in range(images.shape[1])]
    for j, row in enumerate(images):
        for i, image in enumerate(row):
            image_numpy = np.array(image)
            if image_numpy.shape[-1] == 4:
                image_numpy = image_numpy[..., :3]

            transformed_image = base_transform(image=image_numpy)['image']
            with torch.no_grad():
                prediction = model([transformed_image.to(device)])[0]
                nms_prediction = apply_nms(prediction, iou_thresh=0.1)
                bboxes[i][j] = nms_prediction['boxes']
    return bboxes



# def save_model(model, path):
#   torch.save(model, path)

def load_model(name):
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
    model_path = os.path.join(MODELS_PATH, name)
    return torch.load(model_path, map_location)


# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
    final_prediction['scores'] = final_prediction['scores'][keep].cpu()
    final_prediction['labels'] = final_prediction['labels'][keep].cpu()

    return final_prediction


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')


def store_model(title, description, file):
    models_repository.store_model(title, description, file.filename)
    file.save(os.path.join(MODELS_PATH, file.filename))


def get_results_file(source):
    return os.path.join(ANALYZED_PATH, source, 'results.csv')
