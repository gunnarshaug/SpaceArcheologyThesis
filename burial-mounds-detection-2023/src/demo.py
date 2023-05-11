import torch
import argparse
import demo.faster_rcnn

def parse_args():
  parser = argparse.ArgumentParser(description='Predict burial mounds')
  parser.add_argument('--dataset', dest='dataset',
                      help='test dataset',
                      default='datasets/slope/test', type=str)
  parser.add_argument('--config', 
                      help='config file path',
                      default='config/faster_rcnn.yml', type=str)
  parser.add_argument('--model',
                      help='path to model',
                      default='models/replication/Slope_RGB_Fixed_Val_Add_Augm_120_03.03.2023_08.53.39.pt', type=str)
  return parser.parse_args()


def main():
  args = parse_args()
#   config = general.load_cfg(args.config)

  use_cuda = torch.cuda.is_available()

  device = torch.device("cuda" if use_cuda else "cpu")

  model = torch.load(args.model, device)

  image_path = r"C:\Users\egunn\Documents\ArcGIS\Projects\SpaceArcheology\Datasets\Sarpsborg_Halden_2009\Images\Slope_SarpsborgHalden_2009.png"
  result_csv_name = "Composite_Slope_Results_without_Overlap_400"

  store_dir = r"./datasets/experiments"

  img_coordinates = {
    "utm_east": 615371,
    "utm_north": 6575016,
    "pixel_size": 0.5
  }
  #TODO: calculate metrics
  demo.faster_rcnn.Demo(
    model=model,
    device=device,
    store_dir=store_dir,
    img_split_dim={"width":400, "height": 400}
  ).analyze_image(image_path, result_csv_name, **img_coordinates)
    # :param utm_left: UTM coordinate west 
    # :param utm_top: UTM coordinate north
