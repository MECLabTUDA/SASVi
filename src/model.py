import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNNHeads
from torchvision.models.detection.roi_heads import RoIHeads


class CustomRoIHeads(RoIHeads):
    def __init__(self, *args, class_weights=None, **kwargs):
        super(CustomRoIHeads, self).__init__(*args, **kwargs)
        self.class_weights = class_weights

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        """
        Compute the loss for Faster R-CNN.

        Args:
            class_logits (Tensor): predicted class scores
            box_regression (Tensor): predicted box regression deltas
            labels (list[Tensor]): ground-truth class labels
            regression_targets (list[Tensor]): ground-truth box regression targets

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # Flatten labels and regression targets
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # Compute classification loss with class weights
        classification_loss = F.cross_entropy(
            class_logits, labels, weight=self.class_weights
        )

        # Compute box regression loss
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]

        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction='sum',
        ) / labels.numel()

        return classification_loss, box_loss


def get_model_instance_segmentation(num_classes: int,
                                    trainable_backbone_layers: int,
                                    backbone: str = 'ResNet50',
                                    img_size: tuple = (299, 299),
                                    hidden_ft: int | None = None,
                                    custom_in_ft_box: int | None = None,
                                    custom_in_ft_mask: int | None = None,
                                    class_weights: torch.Tensor | None = None) -> nn.Module:

    # Load pre-trained model for instance segmentation
    match backbone.upper():
        case 'RESNET50':
            model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                             weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                                             trainable_backbone_layers=trainable_backbone_layers,
                                             min_size=img_size[0],
                                             max_size=img_size[0])
        case 'RESNET34':
            backbone = resnet_fpn_backbone(backbone_name='resnet34',
                                           weights=ResNet34_Weights.IMAGENET1K_V1,
                                           trainable_layers=trainable_backbone_layers)
            model = MaskRCNN(backbone,
                             num_classes=num_classes,
                             min_size=img_size[0],
                             max_size=img_size[0])

        case 'RESNET18':
            backbone = resnet_fpn_backbone(backbone_name='resnet18',
                                           weights=ResNet18_Weights.IMAGENET1K_V1,
                                           trainable_layers=trainable_backbone_layers)
            model = MaskRCNN(backbone,
                             num_classes=num_classes,
                             min_size=img_size[0],
                             max_size=img_size[0])

        case _:
            raise ValueError(f"backbone '{backbone}' not supported.")

    # Number of input features for the classifier
    if custom_in_ft_box is not None:
        # Create a new box_head with the desired number of features
        representation_size = custom_in_ft_box
        model.roi_heads.box_head = TwoMLPHead(
            in_channels=model.backbone.out_channels * 7 * 7,  # Assuming the default pooling size is 7x7
            representation_size=representation_size
        )
        in_features_box = representation_size
    else:
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    print(f"{in_features_box=}")

    # Replace pre-trained head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box,
                                                      num_classes=num_classes)

    # Number of input features for the mask classifier
    if custom_in_ft_mask is not None:
        # Create a new mask_head with the desired number of features
        model.roi_heads.mask_head = MaskRCNNHeads(
            in_channels=model.backbone.out_channels,
            layers=[256, 256, 256, custom_in_ft_mask],
            dilation=1
        )
        in_features_mask = custom_in_ft_mask
    else:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print(f"{in_features_mask=}")

    if hidden_ft is not None:
        hidden_layer = hidden_ft
    else:
        hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    print(f"{hidden_layer=}")

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                       dim_reduced=hidden_layer,
                                                       num_classes=num_classes)

    if class_weights is not None:

        # Replacing RoIHeads with custom RoIHeads
        model.roi_heads = CustomRoIHeads(
            box_roi_pool=model.roi_heads.box_roi_pool,
            box_head=model.roi_heads.box_head,
            box_predictor=model.roi_heads.box_predictor,
            # Faster R-CNN training
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,  # 256?
            positive_fraction=0.25,  # 0.5?
            bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
            # Faster R-CNN inference
            score_thresh=model.roi_heads.score_thresh,
            nms_thresh=model.roi_heads.nms_thresh,
            detections_per_img=model.roi_heads.detections_per_img,
            # Mask
            mask_roi_pool=model.roi_heads.mask_roi_pool,
            mask_head=model.roi_heads.mask_head,
            mask_predictor=model.roi_heads.mask_predictor,
            # Additional parameters
            keypoint_roi_pool=model.roi_heads.keypoint_roi_pool,
            keypoint_head=model.roi_heads.keypoint_head,
            keypoint_predictor=model.roi_heads.keypoint_predictor,
            # Pass class weights
            class_weights=class_weights
        )

    return model
