from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Dict

class Logger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, image, predicted_boxes, prediction_scores, ground_truth_boxes, location:str=""):        
        raise NotImplementedError

    @abstractproperty
    def timestamp(self):      
        raise NotImplementedError
    
    def info(self, text: str):
        """"
        Used to print information to the console. 
        :param text: the information text.
        """
        print(text)