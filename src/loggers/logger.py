from abc import ABC, abstractmethod
from typing import Optional, Dict

class Logger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        raise NotImplementedError
    