import torch
import utils.metrics
from loggers.logger import Logger
import numpy as np
from abc import abstractproperty, abstractmethod

class BaseTrainer:
    def __init__(self, 
                 device:str,
                 logger: Logger):
        self.device = device
        self.logger = logger
        self.train_loss = utils.metrics.Averager()
        self.val_metrics = utils.metrics.Metrics()
        self.test_metrics = utils.metrics.Metrics()
        self.epochs = self.logger.config["training"]["epochs"]
        self.log_step = int(np.sqrt(self.logger.config["data"]["batch_size"]))
        
        self.optimizer = self.configure_optimizer()
        self.lr_scheduler = self.configure_lr_scheduler()
        
    @abstractmethod
    def _train_step(self, inputs, labels):
        """
        Training logic for one batch
        """
        raise NotImplementedError
    
    @abstractmethod
    def _validate_step(self, inputs, targets):
        """
        Validation logic.
        """
        raise NotImplementedError
    @abstractmethod
    def _test_step(self, inputs, targets):
        """
        Testing logic.
        """
        raise NotImplementedError
    
    @abstractproperty
    def model(self):
        raise NotImplementedError
    
    
    def train(self, data_loaders) -> None:
        """
        Training logic.
        """
        data_loaders.setup("train")
        self.dataloader = data_loaders.train_dataloader()
        self.val_dataloader = data_loaders.val_dataloader()

        for epoch in range(1, self.epochs + 1):
            self.train_loss.reset()      

            self._train_epoch(epoch)
            self.lr_scheduler.step()
            self._validate_epoch(epoch)
            
            print("Test set: Precision: {} Recall: {}\n".format(
                self.val_metrics.precision, 
                self.val_metrics.recall))

            self.logger.log_metrics({
                "Epoch": epoch,
                "Train/Loss": self.train_loss.value,
                "Train/Recall": self.val_metrics.recall,
                "Train/Precision": self.val_metrics.precision,
            })
    def test(self, data_loaders):
        data_loaders.setup("test")
        self.test_dataloader = data_loaders.test_dataloader()
        self.model.eval() 

        for batch_idx, (images, targets ) in enumerate(self.test_dataloader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            boxes = self._test_step(images, targets)
            
            if(batch_idx%self.log_step):       
                self.logger.log_images(images, boxes)
            
        self.logger.log_metrics({
            "Test/NoImages": self.test_metrics.counter,
            "Test/Recall": self.test_metrics.recall,
            "Test/Precision": self.test_metrics.precision,
            "Test/FalsePositives": self.test_metrics.false_positives,
            "Test/FalseNegatives": self.test_metrics.false_negatives,
            "Test/TruePositives": self.test_metrics.true_positives,    
        })
            
            
    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(self.dataloader):
            images = list(image.to(self.device) for image in inputs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
            loss = self._train_step(images, targets)
            
            self.train_loss.update(loss.item())
            
            # compute loss and its gradients
            self.optimizer.zero_grad()
            loss.backward()

            # adjust learning weights based on the gradients we just computed
            self.optimizer.step()
            
            if batch_idx % self.log_step == 0:
                print("Train Epoch: {} {} Loss: {:.6f}".format(
                    epoch,
                    self._progress_text(batch_idx),
                    self.train_loss.value))
                
    def _validate_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_dataloader):
                images = list(image.to(self.device) for image in inputs)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
               
                self._validate_step(images, targets)
                                
    def _progress_text(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx
        total = self.epochs
        return base.format(current, total, 100.0 * current / total)

    def configure_optimizer(self):
        optimizer_config = self.logger.config["training"]["optimizer"]
        if optimizer_config["type"] == "sdg":                
            return torch.optim.SGD(
                params=[p for p in self.model.parameters() if p.requires_grad],
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
             raise ValueError("Invalid optimizer type: {}".format(optimizer_config["type"]))

    def configure_lr_scheduler(self):
        scheduler_config = self.logger.config["training"]["scheduler"]
        if scheduler_config["type"] == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                step_size= scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
                optimizer=self.optimizer
            )
        else:
             raise ValueError("Invalid learning rate scheduler type: {}".format(scheduler_config["type"]))
