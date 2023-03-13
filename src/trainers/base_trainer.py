import torch
from abc import abstractmethod
import os
import utils.general
import utils.model
from utils.metrics import (Averager, Metrics)
import datetime
from loggers.logger import Logger
from abc import abstractproperty
import numpy as np

class BaseTrainer:
    def __init__(self, 
                 config:dict,
                 device:str,
                 logger: Logger,
                 save_model:bool):
        self.device = device
        self.config = config
        self.logger = logger
        self.save_model = save_model
        self.timestamp = datetime.datetime.now().strftime('%d.%m.%Y_%H.%M.%S')
        self.train_loss = Averager()
        self.val_metrics = Metrics()
        self.epochs = self.config['epochs']
        self.log_step = int(np.sqrt(self.config["dataloader"]["batch_size"]))
        self.len_epoch = 0
        
        self.optimizer = self.configure_optimizer()
        self.lr_scheduler = self.configure_lr_scheduler()
    
    @abstractmethod
    def _log_results():
        raise NotImplementedError
    
    @abstractmethod
    def _train_step(self, input, labels):
        """
        Training logic for one batch
        :param batch: Current batch index
        """
        raise NotImplementedError
    
    @abstractmethod
    def _validate_step(self, output, targets):
        """
        Validation logic.
        """
        raise NotImplementedError
    
    def train(self, dataloaders) -> None:
        """
        Training logic.
        """
        dataloaders.setup("train")
        self.dataloader = dataloaders.train_dataloader()
        self.val_dataloader = dataloaders.val_dataloader()
        self.len_epoch = len(self.dataloader)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):
                images = list(image.to(self.device) for image in inputs)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
                loss = self._train_step(images, targets)
                
                self.logger.log_metrics({"train/loss": loss.item()}, batch_idx)

                self.train_loss.update(loss.item())
                
                # compute loss and its gradients
                self.optimizer.zero_grad()
                loss.backward()

                # adjust learning weights based on the gradients we just computed
                self.optimizer.step()
                
                if batch_idx % self.log_step == 0:
                    print('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        self.train_loss.value))
                
            self.lr_scheduler.step()
            self._validate_epoch()
            
            print('Test set: Precision: {} Recall: {}\n'.format(
                self.val_metrics.precision, 
                self.val_metrics.recall))
            

            self.logger.log_metrics(
                {"train/precision": self.val_metrics.precision,
                 "train/recall": self.val_metrics.recall,
                 "train/FP": self.val_metrics.false_positives},
                epoch
            )

        self._log_results()

        if self.save_model:
            self._save_model()
            
    def _validate_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for images, targets in self.val_dataloader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                output = self.model(images)
                self._validate_step(output, targets)

    def _save_model(self) -> None:
        model_dir = self.config["model"]["path"]        
        utils.general.ensure_existing_dir(model_dir)

        path = os.path.join(model_dir,
                            "{}_{}.pt".format(self.config["model"]["name"], self.config["timestamp"]))

        torch.save(self.model,path)
        print("Saving model: {} ...".format(path))
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def configure_optimizer(self):
        optimizer_config = self.config["optimizer"]
        return torch.optim.SGD(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=optimizer_config["lr"],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"]
        )
    def configure_lr_scheduler(self):
        scheduler_config = self.config["scheduler"]
        return torch.optim.lr_scheduler.StepLR(
            step_size= scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
            optimizer=self.optimizer
        )
    
    @abstractproperty
    def model(self):
        raise NotImplementedError