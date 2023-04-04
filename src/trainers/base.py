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
                 config: dict):
        self.device = device
        self.train_loss = utils.metrics.Averager()
        self.val_metrics = utils.metrics.Metrics()
        self.test_metrics = utils.metrics.Metrics()
        self.config = config
        self.epochs = self.config["training"]["epochs"]
        self.log_step = int(np.sqrt(self.config["dataloader"]["batch_size"]))
        self.train_length = 0
        self.checkpoint_dir = self.config.get("model", {}).get("checkpoint_dir", "checkpoints")
        self._optimizer = self._get_optimizer()
        self._lr_scheduler = self._get_lr_scheduler()

        logger_config = self.config["classes"]["logger"]        
        self.logger = utils.general.get_class(**logger_config)(
            **self.config["experiment"],
            config=self.config["training"]
        )
        
    @abstractmethod
    def train_step(self, inputs, labels):
        """
        Training logic for one batch.
        """
        raise NotImplementedError
    
    @abstractmethod
    def validate_step(self, inputs, targets):
        """
        Validation logic for one batch.
        """
        raise NotImplementedError
    @abstractmethod
    def test_step(self, inputs, targets):
        """
        Testing logic for one batch.
        """
        raise NotImplementedError
    
    @abstractmethod
    def on_train_end(self):
        """
        Additional logic after training ends.
        """
        pass
    
    @abstractproperty
    def model(self):
        """
        Returns the deep learning model.
        """
        raise NotImplementedError
    
    def train(self, data_loaders: DataLoaders) -> None:
        """
        Training logic.
        """
        self.logger.log_metrics({
            "NoImages/Train": data_loaders.train_length,
            "NoImages/Val": data_loaders.val_length
        })
        self.train_length = data_loaders.train_length
        
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
            
        self.on_train_end()
        
        
    def test(self, data_loaders: DataLoaders):                
        self.logger.log_metrics({
            "NoImages/Test": data_loaders.test_length
        })
        
        self.model.eval() 
        with torch.no_grad():
            for batch_idx, (images, labels, image_location ) in enumerate(data_loaders.test_dataloader):
                images = list(image.to(self.device) for image in images)
                targets = [{key: value.to(self.device) for key, value in label.items() if not isinstance(value, str)} for label in labels]
                _ = self.test_step(images, targets, image_location)
                                        
            self.logger.log_metrics({
                "Test/Recall": self.test_metrics.recall,
                "Test/Precision": self.test_metrics.precision,
                "Test/F1_Score": self.test_metrics.f1_score,
                "Test/FalsePositives": self.test_metrics.false_positives,
                "Test/FalseNegatives": self.test_metrics.false_negatives,
                "Test/TruePositives": self.test_metrics.true_positives,    
                })
                
            
    def _train_epoch(self, epoch, dataloader):
        self.model.train()
        for batch_idx, (inputs, labels, _) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            images = list(image.to(self.device) for image in inputs)
            targets = [{key: value.to(self.device) for key, value in label.items() if not isinstance(value, str)} for label in labels]
            loss = self.train_step(images, targets)
            
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
            for batch_idx, (inputs, labels, _) in enumerate(dataloader):
                images = list(image.to(self.device) for image in inputs)
                targets = [{key: value.to(self.device) for key, value in label.items() if not isinstance(value, str)} for label in labels]
               
                self.validate_step(images, targets)
                                
    def _train_progress_text(self, batch_index):
        assert self.train_length > 0
        
        base = "[{}/{} ({:.0f}%)]"
        return base.format(batch_index, self.train_length, 100.0 * batch_index / self.train_length)

    @property
    def optimizer(self):
        assert self._optimizer is not None
        return self._optimizer
            
    def _get_optimizer(self):
        optimizer_config = self.config.get("training", {}).get("optimizer", None)
        assert optimizer_config is not None, "[!] optimizer not properly configured"

        supported_optimizers = ("sdg")
        assert optimizer_config["type"] in supported_optimizers, "[!] lr_scheduler must be on of the types: {}".format(supported_optimizers)

        if optimizer_config["type"] == "sdg":
            return torch.optim.SGD(
                params=[p for p in self.model.parameters() if p.requires_grad],
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"]
            )           

    @property
    def lr_scheduler(self):
        assert self._lr_scheduler is not None
        return self._lr_scheduler

    def _get_lr_scheduler(self):
        scheduler_config = self.config.get("training", {}).get("scheduler", None)
        assert scheduler_config is not None, "[!] lr_scheduler not properly configured"
                
        supported_schedulers = ("steplr")
        assert scheduler_config["type"] in supported_schedulers, "[!] lr_scheduler must be on of the types: {}".format(supported_schedulers)
        
        if scheduler_config["type"] == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                step_size= scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
                optimizer=self.optimizer
            )

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pt'
        """
        save_dir = "{}_{}".format(self.checkpoint_dir, self.logger.timestamp)
        utils.general.ensure_existing_dir(save_dir)
        file_path = "{}/checkpoint-epoch{}.pt".format(save_dir, epoch)
        torch.save(self.model, file_path)
        self.logger.info("Saving checkpoint: {} ...".format(file_path))

        if save_best:
            best_path = "{}/model_best.pt".format(save_dir)
            torch.save(self.model, best_path)
            self.logger.info("Saving current best: model_best.pt ...")
