import torch.utils.tensorboard
import torch
import os
import datetime
import argparse
import utils.metrics
import utils.data
import utils.general
import utils.model

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
    config = utils.general.load_cfg(args.config)
  except FileNotFoundError as ex:
    print(ex.strerror, ex.filename)
    return
    
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)
  device = torch.device("cuda" if use_cuda else "cpu")
  print("Using {} device training.".format(device.type))

  train_loader = utils.data.get_dataloader(config, "train")
  val_loader = utils.data.get_dataloader(config, "val")
  test_loader = utils.data.get_dataloader(config, "test")



  model = utils.model.get_pretrained_frcnn()

  model.to(device)
  
  optimizer = torch.optim.SGD(
      params=[p for p in model.parameters() if p.requires_grad],
      lr=config.train.optimizer.lr,
      momentum=config.train.optimizer.momentum,
      weight_decay=config.train.optimizer.weight_decay
  )

  scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size= config.train.scheduler.step_size,
    gamma=config.train.scheduler.gamma
  )

  log_dir = 'tensorboard/frcnn_trainer_{}'.format(config.timestamp)
  comment = 'LR_{}_BATCH_{}' .format(config.train.batch_size, config.train.optimizer.lr)
  tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir, comment=comment)

  train_loss = utils.metrics.Averager()
  val_stats = utils.metrics.Stats()

  for epoch in range(1, config.train.epochs + 1):
    train_loss.reset()
    utils.model.train_one_epoch(model, device, train_loader, optimizer, train_loss)
    scheduler.step()
    utils.model.validate(model, device, val_loader, val_stats)

    tb_writer.add_scalar('Loss/train', 
                          train_loss.value, 
                          epoch)
    tb_writer.add_scalar('Precision/train', val_stats.get_precision(), epoch)
    tb_writer.add_scalar('Recall/train', val_stats.get_recall(), epoch)
    tb_writer.flush()

    print('Train Epoch: {}\tLoss: {}'.format(
        epoch, 
        train_loss.value))
    
    print('Test set: Precision: {} Recall: {}\n'.format(
          val_stats.get_precision(), 
          val_stats.get_recall()))

  if (args.save_model):
    print("saving model..")
    if not os.path.exists(config.model.path):
      print("creating model path: {}".format(config.model.path))
      os.makedirs(config.model.path)

    path = os.path.join(config.model.path,
                        "{}_{}_{}.pt".format(config.model.name, config.train.epochs, config.timestamp))
    torch.save(model,path)

  test_stats = utils.metrics.Stats()
  utils.model.test(config, model, device, test_loader, test_stats, tb_writer)
  # tb_writer.add_scalar('Precision/test', test_stats.get_precision())
  # tb_writer.add_scalar('Recall/test', test_stats.get_recall())

  tb_writer.flush()
  tb_writer.close()

if __name__ == "__main__":
  print(torch.cuda.is_available())
  main()