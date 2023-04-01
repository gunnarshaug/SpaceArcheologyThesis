
from typing import Optional, Any, Dict
import wandb
from loggers.logger import Logger
import datetime

class WandbLogger(Logger):
    def __init__(
        self,
        project: str,
        name: str,
        description: str,
        config:dict=None,
        tags: list=[],
        id: str = None,
        version: str = None,
        anonymous: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`." 
            )
        self._timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        self._wandb_init: Dict[str, Any] = dict(
            project=project,
            notes=description,
            name="{0}_{1}".format(
                name,
                self.timestamp
            ),
            id=version or id,
            resume="allow",
            anonymous=("allow" if anonymous else None),
            tags=tags,
            config=config
        )
        self._wandb_init.update(**kwargs)
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._experiment = None
    @property
    def timestamp(self):
        return self._timestamp
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if step is not None:
            self.experiment.log(metrics, step)
        else:
            self.experiment.log(metrics)

    def log_image(self, image, predicted_boxes, prediction_scores, ground_truth_boxes, location:str=None):
        boxes_predicted =  {
            "predictions":
            {
                "box_data": self._get_bounding_boxes(predicted_boxes, prediction_scores),
            }
        }
        boxes_gt = {
            "ground_truth":
            {
                "box_data": self._get_bounding_boxes(ground_truth_boxes)
            }
        }
        
        wandb_image_predicted = [wandb.Image(image, boxes=boxes_predicted, caption=location if location is not None else "")]
        wandb_image_gt = [wandb.Image(image, boxes=boxes_gt, caption=location if location is not None else "")]

        self.experiment.log({"Images/Predicted": wandb_image_predicted})
        self.experiment.log({"Images/GroundTruth": wandb_image_gt})

    def _get_bounding_boxes(self, boxes, scores=None, class_id=1):

        all_boxes = []
        for idx, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            box_data = {
                "position": {
                    "minX": float(x_min),
                    "maxX": float(x_max),
                    "minY": float(y_min),
                    "maxY": float(y_max)
                },
                "class_id" : class_id,
                "box_caption": format(scores[idx].item(), ".2f") if scores is not None else "",
                "domain" : "pixel"
            }
            all_boxes.append(box_data)
                
        return all_boxes

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