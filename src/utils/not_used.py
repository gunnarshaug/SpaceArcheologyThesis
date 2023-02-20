import os
import cv2
import csv
import torch
import torchvision
import numpy as np
import albumentations as a
import albumentations.pytorch.transforms

#Scan of large Slope image of test area. Set overlap to 0.5 to use overlap scan with 50% percent overlap.
# TODO: cleanup constants, and remove unused code. 
split_width = 400
split_height = 400
image_height = 400
image_width = 400
overlap = 0
data_folder = ""
root = ""
test_folder = ""
device = None


transform = a.Compose([
    a.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    albumentations.pytorch.transforms.ToTensorV2()
])


def start_points(size, split_size):
  points = [0]
  stride = int(split_size * (1 - overlap))
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

def split_image(image):
  img_height, img_width, _ = image.shape
  X_points = start_points(img_width, split_width)
  Y_points = start_points(img_height, split_height)
  
  images = np.empty(shape=(len(X_points), len(Y_points)))
  images = [[[] for i in range(len(X_points))] for i in range(len(Y_points))]
  count = 0

  for i, y_point in enumerate(Y_points):
    for j, x_point in enumerate(X_points):
      split = image[y_point:y_point + split_height, x_point:x_point + split_width]
      images[i][j] = split
      
      #storing

      image_path = os.path.join(
        data_folder,
        "parts",
        'splitted_{}_{}'.format(i, j),
        '.png'
      )
      cv2.imwrite(image_path, split)
      count += 1
  return np.array(images)

def get_bounding_boxes(model, images, transform):
  model.eval()
  bboxes = [[[] for i in range(images.shape[0])] for i in range(images.shape[1])]
  low_scores_count = 0
  for j, row in enumerate(images):
    for i, image in enumerate(row):
      image_numpy = np.array(image)
      transformed_image = transform(image=image_numpy)['image']
      with torch.no_grad():
        prediction = model([transformed_image.to(device)])[0]
        nms_prediction = apply_nms(prediction, iou_thresh=0.1)
        bboxes[i][j] = nms_prediction['boxes']
        for box in nms_prediction['boxes']:
          x_min, y_min, x_max, y_max = box
          cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        image_path = os.path.join(test_folder, 'predictions', '{}_{}.png'.format(i, j))
        cv2.imwrite(image_path, image)
  return bboxes


# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction
  
def normalize_coordinates(bboxes):
  normalized_boxes = []
  for x, row in enumerate(bboxes):
    for y, boxes in enumerate(row):
      x_overlap_padding = overlap * x * split_width
      y_overlap_padding = overlap * y * split_height
      x_padding, y_padding = x * split_width, y * split_height
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

#Saves png image with result bounding boxes.
def save_image(image, bboxes):
  for box in bboxes:
    x_min, y_min, x_max, y_max, label = box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

  image_path = os.path.join(data_folder, 'Composite_Slope_Results_without_Overlap_400.png') 
  cv2.imwrite(image_path, image)



def generate_result_csv(csv_name, bboxes):  
  headers = ['xmin', 'ymin', 'xmax', 'ymax']
  with open(csv_name, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
  
    for box in bboxes:
        xmin, ymin, xmax, ymax, label = box
        writer.writerow([
          xmin.item(), ymin.item(), xmax.item(), ymax.item(),
        ])



def analyze_image(image):
  dir = os.path.join(data_folder, "parts")
  for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
  
  dir = os.path.join(test_folder, "predictions")
  for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

  images = split_image(image)
  bboxes = get_bounding_boxes(
    images, 
    transform= a.Compose([
      a.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      ),
      a.pytorch.transforms.ToTensorV2()
    ])
)
  normalized_bboxes = normalize_coordinates(bboxes)
  save_image(image, normalized_bboxes)
  generate_result_csv(os.path.join(data_folder, 'Composite_Slope_Results_without_Overlap_400.csv'), normalized_bboxes)
  

def main():
  #Scan of large Slope image of test area. Set overlap to 0.5 to use overlap scan with 50% percent overlap.
  image_name = "Rogaland_2016_Composite_Slope.png"
  image_path = os.path.join(root, image_name)
  img = cv2.imread(image_path)

  analyze_image(img)

if __name__ == "__main__":
    main()
