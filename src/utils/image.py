import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_plot(image, bboxes, gt_boxes=None) -> plt.Figure:
    image = image.cpu().permute(1, 2, 0)
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.grid(False)
    ax.axis('off')               

    for box in bboxes["boxes"]:
      x_min, y_min, x_max, y_max = box
      width = x_max - x_min
      height = y_max - y_min
      rect = patches.Rectangle((x_min, y_min), width, height, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
    if gt_boxes != None:
      for box in gt_boxes["boxes"]:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    return fig