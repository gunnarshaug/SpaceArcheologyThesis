import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
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


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
  return torchvision.transforms.ToPILImage()(img).convert('RGB')

def save_parts_with_bb(images, bboxes):
  print(images[0][0])
  for y, row in enumerate(bboxes):
    for x, boxes in enumerate(row):
      pass

def count_boxes(predictions):
  count = 0
  for y, row in enumerate(predictions):
    for x, boxes in enumerate(row):
      count = count + len(boxes)
  print("count_boxes: ", count)

def display_image(image, bboxes, n):
    plt.imshow(image[0])
    ax = plt.gca()
    plt.subplot(2, 2, n)
    plt.grid(False)
    plt.axis('off')               
    plt.rcParams["figure.figsize"] = (1,1)

    for box in bboxes["boxes"]:
      x_min, y_min, x_max, y_max = box
      width = x_max - x_min
      height = y_max - y_min
      rect = patches.Rectangle((x_min, y_min), width, height, edgecolor='r', facecolor='none')
      ax.add_patch(rect)

def resize_bboxes(prediction):
  result = []
  for box in prediction:
    x_min, y_min, x_max, y_max = box
    x1p = x_min / split_width
    x2p = x_max / split_width
    y1p = y_min / split_height
    y2p = y_max / split_height
    
    result.append([
      x1p * image_width,
      y1p * image_height,
      x2p * image_width,
      y2p * image_height
    ])

  return result