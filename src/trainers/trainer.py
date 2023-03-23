import trainers.base_trainer
import torchvision
import utils.model
import utils.metrics
import utils.general
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class Trainer(trainers.base_trainer.BaseTrainer):
    def __init__(self, 
                 device: str,
                 config: dict,
                ):
        self._model = None
        logger_config = config["classes"]["logger"]

        logger = utils.general.get_class(**logger_config)(
            **config["experiment"],
            config=config["training"]
        )
        
        super().__init__(device, logger, config)

    def _train_step(self, inputs, targets):
        output = self.model(inputs, targets)
        losses = sum(loss for loss in output.values())

        return losses
    
    def _validate_step(self, inputs, targets):
        output = self.model(inputs)
        for i, prediction in enumerate(output):
            nms_prediction = utils.model.apply_nms(prediction, iou_thresh=0.1)
            ground_truth = targets[i]

            iou = torchvision.ops.box_iou(
                nms_prediction['boxes'].to(self.device),
                ground_truth['boxes'].to(self.device)
            )

            predicted_boxes_count, gt_boxes_count = list(iou.size())

            if predicted_boxes_count == 0 and gt_boxes_count == 0:
                continue
            
            self.val_metrics.update(iou)
            
    def _test_step(self, inputs, targets):
        output = self.model(inputs)
        boxes = []
        for i, prediction in enumerate(output):
            nms_prediction = utils.model.apply_nms(prediction, iou_thresh=0.1)
            ground_truth = targets[i]
            iou = torchvision.ops.box_iou(
                nms_prediction['boxes'].to(self.device),
                ground_truth['boxes'].to(self.device)
            )
            boxes.append(nms_prediction)
            self.test_metrics.update(iou)
        return boxes

    @property
    def model(self):
        if(self._model is None):
            self._model = get_object_detection_model()              
            self._model.to(self.device)
        return self._model
        
def get_object_detection_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model