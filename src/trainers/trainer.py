import trainers.base_trainer
import torchvision
import utils.model
import utils.metrics
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import loggers.logger 

class Trainer(trainers.base_trainer.BaseTrainer):
    def __init__(self, 
                 config: dict,
                 device: str,
                 logger: loggers.logger.Logger,
                 save_model: bool
                ):
        self._model = None
        super().__init__(config, device, logger, save_model)

    def _train_step(self, images, targets):
        output = self.model(images, targets)
        losses = sum(loss for loss in output.values())
        return losses
    
    def _validate_step(self, output, targets):
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

    def _log_results(self):
        # utils.model.write_results_summary_csv(self.val_metrics, self.config)
        pass

    @property
    def model(self):
        if(self._model is None):
            self._model = get_object_detection_model()              
            self._model.to(self.device)
        return self._model
        
def get_object_detection_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model