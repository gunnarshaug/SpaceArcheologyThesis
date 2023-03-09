from .base_trainer import BaseTrainer
from utils.metrics import (Averager, Stats)
import numpy as np
import torch
import torchvision
from torch.utils.data import (DataLoader)
import utils.model
import utils.metrics

class Trainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 optimizer, 
                 config: dict, 
                 device: str,
                 dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 lr_scheduler=None,
                 save_model:bool=True) -> None:
        super().__init__(model, optimizer, config, save_model)
        self.config = config
        self.device = device
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(dataloader.batch_size))
        self.len_epoch = len(self.dataloader)
        self.train_loss = Averager()
        self.val_metrics = Stats()

    def _train_epoch(self, epoch) -> None:
        self.model.train()
        self.train_loss.reset()

        for batch_idx, (inputs, labels) in enumerate(self.dataloader):
            images = list(image.to(self.device) for image in inputs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
            output = self.model(images, targets)

            losses = sum(loss for loss in output.values())
            loss_value = losses.item()

            self.train_loss.update(loss_value)

            # compute loss and its gradients
            self.optimizer.zero_grad()
            losses.backward()
            # adjust learning weights based on the gradients we just computed
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.train_loss.value))
                
        self.lr_scheduler.step()
        self._validate_epoch(epoch)

        self.logger.debug('Test set: Precision: {} Recall: {}\n'.format(
            self.val_metrics.get_precision(), 
            self.val_metrics.get_recall()))

        self.writer.add_scalar('Precision/train', self.val_metrics.get_precision(), epoch)
        self.writer.add_scalar('Recall/train', self.val_metrics.get_recall(), epoch)
        self.writer.add_scalar('FalsePositives/train', self.val_metrics.get_false_positives(), epoch)


    def _validate_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            for images, targets in self.val_dataloader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                output = self.model(images)
                for i, result in enumerate(output):
                    nms_prediction = utils.model.apply_nms(result, iou_thresh=0.1)
                    ground_truth = targets[i]
                    iou = torchvision.ops.box_iou(
                        nms_prediction['boxes'].to(self.device),
                        ground_truth['boxes'].to(self.device)
                    )
                    predicted_boxes_count, gt_boxes_count = list(iou.size())

                    if predicted_boxes_count == 0 and gt_boxes_count == 0:
                        continue
                    
                    self.val_metrics.update(iou)
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
        
    def _log_results(self):
        utils.model.write_results_summary_csv(self.val_metrics, self.config)
        