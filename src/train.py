import torchvision
import torchvision.models.detection as m
import argparse
import torch
import os
import utils.metrics
import utils.cnf
import utils.data
import utils.general
import matplotlib as plt

def train(args, model, device, train_loader, optimizer, epoch):
  loss_hist = utils.metrics.Averager()
  loss_hist.reset()
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    # data, target = data.to(device), target.to(device)
    images = list(image.to(device) for image in data)
    targets = [{k: v.to(device) for k, v in t.items()} for t in target]
    # print("targets: {}".format(targets))

    # output = model(data, target)
    output = model(images, targets)
    losses = sum(loss for loss in output.values())
    loss_value = losses.item()

    loss_hist.send(loss_value)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss_value))
      print("Loss from hist: ", loss_hist.value)



def validate(model, device, val_loader):
  model.eval() 
  stats = utils.metrics.Stats()
  with torch.no_grad():
    for images, targets in val_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      output = model(images)
      for i, result in enumerate(output):
        nms_prediction = utils.general.apply_nms(output[i], iou_thresh=0.1)
        ground_truth = targets[i]
        iou = torchvision.ops.box_iou(nms_prediction['boxes'].to(device), ground_truth['boxes'].to(device))
        predicted_boxes_count, gt_boxes_count = list(iou.size())

        if predicted_boxes_count == 0 and gt_boxes_count == 0:
          continue
        
        tp, fp, fn = utils.metrics.compute_accuracy(iou)
        stats.send(tp, fp, fn)

  print('\nTest set: Precision: {},\t Recall: {}\n'.format(
      stats.get_precision(), stats.get_recall()))


def parse_args():
  parser = argparse.ArgumentParser(description='Pytorch Faster R-CNN')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=True,
                      help='For Saving the current Model')
  parser.add_argument('--config',
                      help='location of config file',
                      default='config/faster_rcnn.yml', type=str)
  return parser.parse_args()


def main():
  args = parse_args()
  try:
    config = utils.cnf.load_cnf(args.config)
  except FileNotFoundError as ex:
    print(ex.strerror, ex.filename)
    return
    
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)
  device = torch.device("cuda" if use_cuda else "cpu")
  print("Using {} device training.".format(device.type))

  train_loader = utils.data.create_dataloader(config, "train")
  val_loader = utils.data.create_dataloader(config, "val")

  model = m.fasterrcnn_resnet50_fpn(weights=m.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

  model.to(device)
  
  optimizer = torch.optim.SGD(
      params=[p for p in model.parameters() if p.requires_grad],
      lr=config.train.optimizer.lr,
      momentum=config.train.optimizer.stg.momentum,
      weight_decay=config.train.optimizer.stg.weight_decay
  )

  scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size= config.train.optimizer.stg.scheduler.step_size,
    gamma=config.train.optimizer.stg.scheduler.gamma
  )

  learning_rates = []
  for epoch in range(1, config.train.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    learning_rates.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

    validate(model, device, val_loader)
  plt.plot(range(config.train.epochs),learning_rates)

  if (args.save_model):
    print("saving model..")
    if not os.path.exists(config.model.path):
      print("creating model path: {}".format(config.model.path))
      os.makedirs(config.model.path)

    path = os.path.join(config.model.path, config.model.name)
    torch.save(model,path)
    # st_path = os.path.join(config.model.path, "st_" + config.model.name)
    # torch.save(model.state_dict(), st_path)

if __name__ == "__main__":
  print(torch.cuda.is_available())
  # main()
  # model = m.fasterrcnn_resnet50_fpn(weights=m.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

