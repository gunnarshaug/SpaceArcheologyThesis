import torch
import utils.metrics
import utils.general
from loggers.logger import Logger
import numpy as np
from abc import abstractproperty, abstractmethod
from data.dataloaders import DataLoaders

class BaseTrainer:
    def __init__(self,
                 device:str,
                 logger: Logger,
                 config: dict,
                 checkpoint_dir: str):
        self.device = device
        self.logger = logger
        self.train_loss = utils.metrics.Averager()
        self.val_metrics = utils.metrics.Metrics()
        self.test_metrics = utils.metrics.Metrics()
        self.config = config
        self.epochs = self.config["training"]["epochs"]
        self.log_step = int(np.sqrt(self.config["dataloader"]["batch_size"]))
        self.train_length = 0
        self.checkpoint_dir = checkpoint_dir
        
        # self.log_image_batch_idx = np.random.randint(0, self.logger["batch_size"])
        self.log_image_batch_idx = 0        
    @abstractmethod
    def _train_step(self, inputs, labels):
        """
        Training logic for one batch.
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
    
    def train(self, data_loaders: DataLoaders) -> None:
        """
        Training logic.
        """

        for epoch in range(1, self.epochs + 1):
            self.train_loss.reset()      

            self._train_epoch(epoch, data_loaders.train_dataloader)
            self._validate_epoch(epoch, data_loaders.val_dataloader)
            
            self.lr_scheduler.step()
            
            
            self.logger.info("Val set: Precision: {} Recall: {}\n".format(
                self.val_metrics.precision, 
                self.val_metrics.recall))

            self.logger.log_metrics({
                "Epoch": epoch,
                "Train/Loss": self.train_loss.value,
                "Train/Recall": self.val_metrics.recall,
                "Train/Precision": self.val_metrics.precision,
            })
            
            self._save_checkpoint(epoch, save_best=self.val_metrics.is_improving)
        
    def test(self, data_loaders: DataLoaders):                
        self.model.eval() 

        for batch_idx, (images, targets ) in enumerate(data_loaders.test_dataloader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            boxes = self._test_step(images, targets)
                        
            if batch_idx == self.log_image_batch_idx and boxes is not None:
                for idx, image in enumerate(images):     
                    self.logger.log_image(image, boxes[idx], targets[idx])
            
        self.logger.log_metrics({            "Test/NoImages": self.test_metrics.counter,
            "Test/Recall": self.test_metrics.recall,
            "Test/Precision": self.test_metrics.precision,
            "Test/FalsePositives": self.test_metrics.false_positives,
            "Test/FalseNegatives": self.test_metrics.false_negatives,
            "Test/TruePositives": self.test_metrics.true_positives,    
        })
            
            
    def _train_epoch(self, epoch, dataloader):
        self.model.train()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            images = list(image.to(self.device) for image in inputs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
            loss = self._train_step(images, targets)
            
            self.train_loss.update(loss.item())
            
            # compute loss and its gradients
            loss.backward()

            # adjust learning weights based on the gradients we just computed
            self.optimizer.step()
            
            if (batch_idx+1) % self.log_step == 0:
                self.logger.info("Train Epoch: {} {} Loss: {:.6f}".format(
                    epoch,
                    self._train_progress_text(batch_idx+1),
                    self.train_loss.value))
                
             
    def _validate_epoch(self, epoch, dataloader):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                images = list(image.to(self.device) for image in inputs)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
               
                self._validate_step(images, targets)
                                
    def _train_progress_text(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx
        return base.format(current, self.train_length, 100.0 * current / self.train_length)

    @property
    def optimizer(self):
        optimizer_config = self.config["training"]["optimizer"]
        if optimizer_config["type"] == "sdg":                
            return torch.optim.SGD(
                params=[p for p in self.model.parameters() if p.requires_grad],
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
             raise ValueError("Invalid optimizer type: {}".format(optimizer_config["type"]))

    @property
    def lr_scheduler(self):
        scheduler_config = self.config["training"]["scheduler"]
        if scheduler_config["type"] == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                step_size= scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
                optimizer=self.optimizer
            )
        else:
             raise ValueError("Invalid learning rate scheduler type: {}".format(scheduler_config["type"]))

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pt'
        """
        
        utils.general.ensure_existing_dir(self.checkpoint_dir)
        file_path = "{}/checkpoint-epoch{}.pt".format(self.checkpoint_dir,epoch)
        torch.save(self.model, file_path)
        self.logger.info("Saving checkpoint: {} ...".format(file_path))

        if save_best:
            best_path = "{}/model_best.pt".format(self.checkpoint_dir)
            torch.save(self.model, best_path)
            self.logger.info("Saving current best: model_best.pt ...")
