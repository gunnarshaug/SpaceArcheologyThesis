import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import Averager, Stats, compute_accuracy,apply_nms
from config import device, train_loader, val_loader, data_folder, test_loader,display_image
import os
import torch
from evaluate import evaluate_model

def get_object_detection_model(num_classes = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def train_model(model, train_loader, val_loader, optimizer, num_epochs):
  loss_history = []
  train_history = []
  val_history = []

  loss_hist = Averager()
  train_hist = Averager()
  val_hist = Averager()

  for epoch in range(num_epochs):
    print("Epoch: ", epoch + 1)

    model.train()

    loss_hist.reset()
    train_hist.reset()
    val_hist.reset()

    for images, targets in train_loader:
      display_image(images[1], targets[0])
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      loss_dict = model(images, targets)

      losses = sum(loss for loss in loss_dict.values())
      loss_value = losses.item()

      loss_hist.send(loss_value)

      optimizer.zero_grad()
      losses.backward()
      optimizer.step()
    
    evaluate(model, val_loader)
    print("Loss: ", loss_hist.value)  

def evaluate(model, loader):
    model.eval() # Evaluation mode
    stats = Stats()
    with torch.no_grad():
        for images, targets in loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            results = model(images)
            for i, result in enumerate(results):
              nms_prediction = apply_nms(results[i], iou_thresh=0.1)
              ground_truth = targets[i]
              iou = torchvision.ops.box_iou(nms_prediction['boxes'].to(device), ground_truth['boxes'].to(device))
              predicted_boxes_count, gt_boxes_count = list(iou.size())

              if predicted_boxes_count == 0 and gt_boxes_count == 0:
                continue
              
              tp, fp, fn = compute_accuracy(iou)
              stats.send(tp, fp, fn)


    print("Precision: ", stats.get_precision())
    print("Recall: ", stats.get_recall())

def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #lr_scheduler = None

    if torch.cuda.is_available():
        model.cuda()


    train_model(model, train_loader, val_loader, optimizer, 1)
    # model_file = os.path.join(data_folder, "model_120_no_optim_slope_best.pt")
    model_file = os.path.join("C:/Bachelor", "model_120_no_optim_slope_best.pt")
    torch.save(model, model_file)

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()