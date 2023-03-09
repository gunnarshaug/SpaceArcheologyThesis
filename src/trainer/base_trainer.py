import torch
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
import utils.general
import utils.model
import torchvision
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')

class BaseTrainer:
    def __init__(self, model, optimizer, config:dict, save_model:bool):
        self.config = config

        self.logger = logging.getLogger('trainer')

        self.model = model
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']

        log_dir = 'tensorboard/frcnn_trainer_{}'.format(config["timestamp"])
        self.writer = SummaryWriter(log_dir)
        self.early_stop = np.inf
        self.save_model = save_model


    @abstractmethod
    def _log_results():
        raise NotImplementedError
    
    @abstractmethod
    def _train_epoch(self, epoch:int) -> None:
        """
        Training logic for one epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    def train(self) -> None:
        """
        Training logic for all epochs
        """

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        self.writer.flush()
        self.writer.close()

        self._log_results()

        if self.save_model:
            self._save_model()

    def _save_model(self) -> None:
        model_dir = self.config["model"]["path"]        
        utils.general.ensure_existing_dir(model_dir)

        path = os.path.join(model_dir,
                            "{}_{}.pt".format(self.config["model"]["name"], self.config["timestamp"]))

        torch.save(self.model,path)
        self.logger.info("Saving model: {} ...".format(path))