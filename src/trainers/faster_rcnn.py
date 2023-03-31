import trainers.base
import torchvision
import utils.metrics
import utils.general
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Trainer(trainers.base.BaseTrainer):
    def __init__(self, 
                 device: str,
                 config: dict):
        
        self.image_log_count = 0
        self._model = None
                
        checkpoint_dir = config.get("model", {}).get("checkpoint_dir", "checkpoints")
        
        super().__init__(device, config, checkpoint_dir=checkpoint_dir)

    def train_step(self, inputs, targets):
        output = self.model(inputs, targets)
        losses = sum(loss for loss in output.values())

        return losses
    
    def validate_step(self, inputs, targets):
        output = self.model(inputs)
        for i, prediction in enumerate(output):
            nms_prediction = utils.general.apply_nms(prediction, iou_thresh=0.1)
            ground_truth = targets[i]

            iou = torchvision.ops.box_iou(
                nms_prediction['boxes'].to(self.device),
                ground_truth['boxes'].to(self.device)
            )

            predicted_boxes_count, gt_boxes_count = list(iou.size())

            if predicted_boxes_count == 0 and gt_boxes_count == 0:
                continue
            
    def test_step(self, inputs, targets):
        output = self.model(inputs)
        boxes = []
        for i, prediction in enumerate(output):
            nms_prediction = utils.general.apply_nms(prediction, iou_thresh=0.1)
            ground_truth = targets[i]
            
            # IoU measures the overlap between two bounding boxes.
            iou = torchvision.ops.box_iou(
                nms_prediction['boxes'].to(self.device),
                ground_truth['boxes'].to(self.device)
            )
            boxes.append(nms_prediction)
            self.test_metrics.update(iou)
            
            self.val_metrics.update(iou)
            
            if self.image_log_count <= 10: 
                self.image_log_count += 1
                parameters = {
                    "image": inputs[i],
                    "predicted_boxes":  nms_prediction["boxes"],
                    "ground_truth_boxes": ground_truth["boxes"],
                    "prediction_scores":nms_prediction["scores"]
                }
                self.has_logged_image = True
                self.logger.log_image(**parameters)
            
        return boxes

    @property
    def model(self):
        if(self._model is None):
            self._model = _get_object_detection_model()              
            self._model.to(self.device)
        return self._model
        
def _get_object_detection_model(num_classes: int=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model