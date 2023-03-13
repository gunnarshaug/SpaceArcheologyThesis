from abc import ABC, abstractmethod
from typing import Optional, Dict

class Logger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        raise NotImplementedError