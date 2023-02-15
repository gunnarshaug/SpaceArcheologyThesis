import torchvision
import argparse
import torch
import utils.config
import utils.data
import utils.test
import utils.metrics
import utils.model

def train(args, model, device, train_loader, optimizer, epoch):
  loss_hist = utils.metrics.Averager()
  loss_hist.reset()
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    # data, target = data.to(device), target.to(device)
    images = list(image.to(device) for image in data)
    targets = [{k: v.to(device) for k, v in t.items()} for t in target]
    print("targets: {}".format(targets))

    # output = model(data, target)
    output = model(images, targets)
    losses = sum(loss for loss in output.values())
    loss_value = losses.item()

    loss_hist.send(loss_value)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss_value))
    # if batch_idx % args.log_interval == 0:
        # print("Loss: ", loss_hist.value)  


def validate(model, device, val_loader):
  model.eval() 
  stats = utils.metrics.Stats()
  with torch.no_grad():
    for images, targets in val_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      output = model(images)
      for i, result in enumerate(output):
        nms_prediction = utils.test.apply_nms(output[i], iou_thresh=0.1)
        ground_truth = targets[i]
        iou = torchvision.ops.box_iou(nms_prediction['boxes'].to(device), ground_truth['boxes'].to(device))
        predicted_boxes_count, gt_boxes_count = list(iou.size())

        if predicted_boxes_count == 0 and gt_boxes_count == 0:
          continue
        
        tp, fp, fn = utils.metrics.compute_accuracy(iou)
        stats.send(tp, fp, fn)

  print('\nTest set: Precision: {},\t Recall: {}\n'.format(
      stats.get_precision(), stats.get_recall()))


def parse_args(config):
  parser = argparse.ArgumentParser(description='Pytorch Faster R-CNN')
  parser.add_argument('--batch-size', 
                        type=int, 
                        default=config.train.batch_size,
                        metavar='N',
                        help='input batch size for training (default: {})'.format(
                          config.train.batch_size)
                      )
  parser.add_argument('--test-batch-size', 
                        type=int, 
                        default=config.test.batch_size,
                        metavar='N',
                        help='input batch size for testing (default: {})'.format(
                          config.test.batch_size)
                        )
  parser.add_argument('--epochs', type=int, 
                        default=config.train.epochs,
                        metavar='N',
                        help='number of epochs to train (default: {})'.format(
                          config.train.epochs
                        ))
  parser.add_argument('--lr', type=float,
                        default=config.train.optimizer.initial_lr,
                        metavar='LR',
                        help='learning rate (default: {})'.format(
                          config.train.optimizer.initial_lr
                        ))
  parser.add_argument('--momentum', type=float, 
                        default=config.train.optimizer.stg.momentum,
                        metavar='M',
                        help='SGD momentum (default: {})'.format(
                          config.train.optimizer.stg.momentum
                        ))
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')
  # parser.add_argument('--cfg', dest='cfg_file',
  #                     help='optional config file',
  #                     default='config/faster_rcnn.yml', type=str)
  return parser.parse_args()

def main():
  config = utils.config.load("config/faster_rcnn.yml")
  args = parse_args(config)
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)
  device = torch.device("cuda" if use_cuda else "cpu")
  print("Using {} device training.".format(device.type))

  train_loader = utils.data.create_dataloader(config)
  val_loader = utils.data.create_dataloader(config, is_train=False)

  model = utils.model.get_object_detection_model()
  model.to(device)
  
  optimizer = utils.data.make_optimizer(config, model)

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    validate(model, device, val_loader)

  if (args.save_model):
    torch.save(model.state_dict(),config.model.name)

if __name__ == "__main__":
  print(torch.cuda.is_available())

  main()