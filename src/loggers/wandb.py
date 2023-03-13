
from typing import Optional, Any, Dict
import wandb
from loggers.logger import Logger

class WandbLogger(Logger):
    def __init__(
        self,
        name: str = None,
        save_dir:str = ".",
        version: str = None,
        dir: str = None,
        id: str = None,
        anonymous: Optional[bool] = None,
        project: str = "lightning_logs",
        **kwargs: Any,
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`." 
            )

        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = dict(
            name=name,
            project=project,
            dir=save_dir or dir,
            id=version or id,
            resume="allow",
            anonymous=("allow" if anonymous else None),
        )
        self._wandb_init.update(**kwargs)
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._experiment = None
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:

        if step is not None:
            self.experiment.log(metrics, step)
        else:
            self.experiment.log(metrics)
            
    @property
    def experiment(self):
        """
        Wandb object.
        """
        if self._experiment is None:
            if wandb.run is not None:
                print(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            else:
                self._experiment = wandb.init(**self._wandb_init)

        return self._experiment