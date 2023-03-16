
from typing import Optional, Any, Dict
import wandb
from loggers.logger import Logger
import datetime

class WandbLogger(Logger):
    def __init__(
        self,
        config:dict,
        version: str = None,
        save_dir: str = None,
        dir: str = None,
        id: str = None,
        anonymous: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`." 
            )

        self.config = config
        self.timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        logger_config = self.config["logger"]
        self._wandb_init: Dict[str, Any] = dict(
            project=logger_config["experiment_name"],
            notes=logger_config["experiment_description"],
            name="run_{0}_{1}".format(
                self.timestamp,
                logger_config["dtm_vs_technique"]
            ),
            dir=save_dir or dir,
            id=version or id,
            resume="allow",
            anonymous=("allow" if anonymous else None),
            tags=[logger_config['dtm_vs_technique']],
            config=self.config
        )
        self._wandb_init.update(**kwargs)
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._experiment = None
        self.class_id_to_label = {
            1: "mound",
            2: "background"
        }
            
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if step is not None:
            self.experiment.log(metrics, step)
        else:
            self.experiment.log(metrics)

    def log_images(self, images, bboxes, key="images"):

        wandb_images = [self._generate_bounding_boxes(image, bboxes) for image in images]
        # wandb_images = [wandb.Image(image, caption="Top: Output, Bottom: SLOPE Image") for image in images]
        self.experiment.log({str(key): wandb_images})

    def _generate_bounding_boxes(self, image, bboxes_list):

        all_boxes = []
        for boxes in bboxes_list:
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                box_data = {
                    "position": {
                        "minX": float(x_min),
                        "maxX": float(x_max),
                        "minY": float(y_min),
                        "maxY": float(y_max)
                    },
                    "class_id" : 1,
                    "box_caption": "mound",
                    "domain" : "pixel"
                }
                all_boxes.append(box_data)
                
        box_image = wandb.Image(image, boxes={"predictions": {"box_data": all_boxes, "class_labels" : self.class_id_to_label}})
        return box_image

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
                self._experiment.define_metric("Train/Loss", summary="min")
                self._experiment.define_metric("Train/Precision", summary="max")
                self._experiment.define_metric("Train/Recall", summary="max")

        return self._experiment