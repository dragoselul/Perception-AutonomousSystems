import pytorch_lightning as pl
import torch
import torchvision
import sys
from torchvision.ops import nms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision



def NMS(predictions, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes
    NMS is applied as a post-processing step to remove redundant bounding boxes (only in inference mode)
    :param predictions: Dictionary containing bounding boxes, scores, and labels
    :param iou_threshold: IoU threshold for NMS
    :return: Filtered predictions after applying NMS
    """
    
    boxes = predictions["boxes"]    # List of bounding boxes
    scores = predictions["scores"]  # List of scores for each bounding box
    labels = predictions["labels"]  # List of labels for each bounding box
    
    # Retrieve the indices of the boxes to keep after applying NMS
    nms_indices = nms(boxes, scores, iou_threshold)
    
    return {
        "boxes": boxes[nms_indices],
        "scores": scores[nms_indices],
        "labels": labels[nms_indices],
    }
    

class YoloLightning(pl.LightningModule):
    
    def __init__(self, num_classes, learning_rate=1e-4, pretrained=True):
        super().__init__()
        
        # Initialize the YOLOv5 model from torchvision
        model = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', 
            pretrained=pretrained)
        
        self.compute_loss = self.model.compute_loss
        self.learning_rate = learning_rate
        
        
    def _shared_step(self, batch, stage='train'):
        imgs, targets = batch          # adjust if your dataloader is different

        # Forward pass: YOLOv5 returns predictions as list of tensors
        preds = self.model(imgs)

        # Compute loss using YOLO's internal loss function
        loss, loss_items = self.compute_loss(preds, targets)

        # Log each loss component (box, obj, cls)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss_box", loss_items[0], on_epoch=True)
        self.log(f"{stage}_loss_obj", loss_items[1], on_epoch=True)
        self.log(f"{stage}_loss_cls", loss_items[2], on_epoch=True)

        return loss
        
        
    def training_step(self, batch, _):
        return self._shared_step(batch, stage='train')
    
    def validation_step(self,  batch, _):
        return self._shared_step(batch, stage='val')
    
    def test_step(self, batch, _):
        return self._shared_step(batch, stage='test')
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler
        :return: Dictionary containing the optimizer and learning rate scheduler
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
        
class FasterRCNNLightning(pl.LightningModule):
    """
    Lightning module for the Faster R-CNN model
    """

    def __init__(self, num_classes, learning_rate=1e-4, pretrained=True):
        super().__init__()
        """ 
        Initializes the Faster R-CNN model lightning module
        :param num_classes: Number of classes in the dataset
        :param learning_rate: Learning rate for the optimizer
        """
        
        self.learning_rate = learning_rate
        
        # Load the Faster R-CNN model as the backbone of the Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")  
        # Get the number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )  
        
        # Initialize the metrics
        self.val_map_metric = MeanAveragePrecision()
        self.val_iou_metric = IntersectionOverUnion()
        self.test_map_metric = MeanAveragePrecision()
        self.test_iou_metric = IntersectionOverUnion()

        torch.set_float32_matmul_precision("high")

    def training_step(self, batch, _):
        """
        Training step for the model
        :param batch: Tuple containing images and targets
        :param _: Batch index (not used)
        :return: Dictionary containing the loss
        """
        
        images, targets = batch
        # Forward pass with targets
        loss_dict = self.model(images, targets)  
        # Sum of all losses (classification and regression)
        total_loss = sum(loss for loss in loss_dict.values())  
        
        # Log all the losses as well as the total loss
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return {"loss": total_loss}
    
    def validation_step(self, batch, _):
        """
        Validation step for the model
        :param batch: Tuple containing images and targets
        :param _: Batch index (not used)
        :return: Dictionary containing the loss, images, predictions, and targets
        """
        
        # Log the evaluation metrics
        self.model.eval()

        images, targets = batch
        # Forward pass without targets
        predictions = self.model(images) 
        
        # Update the mAP metric and compute the metrics
        self.val_map_metric.update(predictions, targets)
        val_metrics_map = self.val_map_metric.compute()
        for k, v in val_metrics_map.items():
            if isinstance(v, torch.Tensor):
                # Do not log the classes info (it's not a metric, just list of predicted classes)
                if k != "classes":
                    value = v.mean() if v.numel() > 0 else torch.tensor(0.0)
                    # Log the metric
                    self.log(f"val_{k}", value, prog_bar=True)
            else:
                # Log the metric
                self.log(f"val_{k}", v, prog_bar=True)
        
        # Update the IoU metric
        self.val_iou_metric.update(predictions, targets)
        self.log_dict({f"val_{k}": v for k, v in self.val_iou_metric.compute().items()}, prog_bar=True)
        
        # Log the validation loss
        self.model.train()
        # Forward pass with targets
        loss_dict = self.model(images, targets)  
        total_loss = sum(loss for loss in loss_dict.values())
        self.log("val_total_loss", total_loss, prog_bar=True)

        return {
            "loss": total_loss,
            "images": images,
            "predictions": predictions,
            "targets": targets,
        }

    def test_step(self, batch, _):
        """
        Test step
        :param batch: Tuple containing images and targets
        :param batch_idx: Index of the batch
        """

        images, targets = batch
        # Forward pass without targets
        predictions = self.model(images)
        
        # Update the mAP metric
        self.test_map_metric.update(predictions, targets)
        test_metrics_map = self.test_map_metric.compute()
        for k, v in test_metrics_map.items():
            if isinstance(v, torch.Tensor):
                # Do not log the classes info (it's not a metric, just list of predicted classes)
                if k != "classes":
                    value = v.mean() if v.numel() > 0 else torch.tensor(0.0)
                    # Log the metric
                    self.log(f"test_{k}", value, prog_bar=True)
            else:
                # Log the metric
                self.log(f"test_{k}", v, prog_bar=True)
        
        # Update the IoU metric
        self.test_iou_metric.update(predictions, targets)
        self.log_dict({f"test_{k}": v for k, v in self.test_iou_metric.compute().items()}, prog_bar=True)

        return {
            "images": images,
            "predictions": predictions,
            "targets": targets,
        }

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler
        :return: Dictionary containing the optimizer and learning rate scheduler
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
    
    
if __name__ == "__main__":
    model = YoloLightning(num_classes=2)
    print(model)
    
    # Dummy input
    imgs = torch.randn(2, 3, 224, 224)
    preds = model.model(imgs)
    print(preds)

