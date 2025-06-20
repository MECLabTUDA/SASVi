import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import Dataset
from transformers import (DetrImageProcessor, DetrForSegmentation, Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor)

from sds_playground.datasets import CaDISv2_Dataset, CholecSeg8kDataset, Cataract1kSegmentationDataset
from sds_playground.utils import convert_to_integer_mask, convert_mask_to_RGB, denormalize

from src.model import get_model_instance_segmentation
from src.data import insert_component_masks, remap_labels, remove_mask_overlap


def create_logger(log_file_path: Path):
    # Create a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)  # Set the level of the logger

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def visualise_results(images_list: list,
                      targets: list,
                      outputs: list,
                      ds: Dataset,
                      ignored_indices: list,
                      shift_by_1: bool,
                      exp_dir: Path,
                      remove_overlap: bool = False,
                      target_size: tuple | None = None,
                      device: str = 'cuda',
                      img_norm: tuple = (.5, .5),
                      max_img_count: int = 8,
                      name: str = 'val_examples.png'
                      ):

    """ Visualises object detection and segmentation results in a plot.

        TODO: Resize images, masks and bbs to the original size

    :param images_list: List of input image tensors in (3, H, W)
    :param targets: Target dictionary; containing masks, boxes, labels
    :param outputs: Prediction dictionary; containing masks, boxes, labels
    :param ds: Dataset
    :param ignored_indices: List of images that were ignored for training the model
    :param shift_by_1:
    :param exp_dir: Target directory for saving the plot
    :param remove_overlap: Remove overlap from binary segmentation masks, favoring higher indices
    :param target_size: Target size for resizing images, masks and boxes (default None)
    :param device: Device literal
    :param img_norm: Img norm (mean, std) that was used to normalise the images for training the model
    :param max_img_count: Maximum number of images to display
    :param name: File name for saving the plot
    :return:
    """

    N = min(len(targets), max_img_count)
    fig, ax = plt.subplots(nrows=5, ncols=N, figsize=(3 * N, 10))

    ax[0, 0].set_ylabel('Image')
    ax[1, 0].set_ylabel('GT BBs')
    ax[2, 0].set_ylabel('PRed. BBs')
    ax[3, 0].set_ylabel('GT Mask')
    ax[4, 0].set_ylabel('Pred. Masks')

    for n in range(N):

        # Input image
        img = denormalize(images_list[n], *img_norm)
        if target_size is not None:
            original_height, original_width = img.shape[1:]
            img = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            scale_y, scale_x = target_size[0] / original_height, target_size[1] / original_width
        ax[0, n].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax[0, n].set_xticks([])
        ax[0, n].set_yticks([])

        # GT BB
        ax[1, n].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        for i, box in enumerate(targets[n]["boxes"]):
            xmin, ymin, xmax, ymax = box.cpu()
            if target_size is not None:
                xmin, xmax = xmin * scale_x, xmax * scale_x
                ymin, ymax = ymin * scale_y, ymax * scale_y
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r',
                                     facecolor='none')
            ax[1, n].add_patch(rect)
            # TODO: Add label names or coloring based on palette
            # plt.text(xmin, ymin, f'{labels[i]}', color='white', fontsize=12,
            #         bbox=dict(facecolor='red', alpha=0.5))
        ax[1, n].set_xticks([])
        ax[1, n].set_yticks([])

        # BB pred
        ax[2, n].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        for i, box in enumerate(outputs[n]["boxes"]):
            xmin, ymin, xmax, ymax = box.cpu()
            if target_size is not None:
                xmin, xmax = xmin * scale_x, xmax * scale_x
                ymin, ymax = ymin * scale_y, ymax * scale_y
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r',
                                     facecolor='none')
            ax[2, n].add_patch(rect)
            # TODO: Add label names or coloring based on palette
            #plt.text(xmin, ymin, f'{labels[i]}', color='white', fontsize=12,
            #         bbox=dict(facecolor='red', alpha=0.5))
        ax[2, n].set_xticks([])
        ax[2, n].set_yticks([])

        # GT mask
        # Re-map labels shifted by 1 (since 0 is BG in Mask R-CNN)
        if shift_by_1:
            remapped_gt_labels = targets[n]['labels'] - 1
        else:
            remapped_gt_labels = targets[n]['labels']
        # Re-map labels (from reduced label set) for coloring / visualisation
        remapped_gt_labels = remap_labels(remapped_gt_labels, ds.num_classes, ignored_indices)
        padded_gt_mask = insert_component_masks(
            targets[n]["masks"],
            remapped_gt_labels,
            ds.num_classes,  # num_classes + 1 if ignored_indices[0] is not None else num_classes,
            ignore_index=ds.ignore_index
        ).to(device)
        if remove_overlap:
            padded_gt_mask = remove_mask_overlap(padded_gt_mask.unsqueeze(0)).squeeze(0)
        if target_size is not None:
            padded_gt_mask = F.interpolate(padded_gt_mask.unsqueeze(0).float(), size=target_size, mode='nearest').squeeze(0)

        int_gt_mask = convert_to_integer_mask(padded_gt_mask.unsqueeze(0),
                                              num_classes=ds.num_classes,
                                              ignore_index=ds.ignore_index)

        rgb_gt_mask = convert_mask_to_RGB(int_gt_mask, palette=ds.get_cmap().to(device), ignore_index=ds.ignore_index)
        rgb_gt_mask = rgb_gt_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax[3, n].imshow(rgb_gt_mask)
        ax[3, n].set_xticks([])
        ax[3, n].set_yticks([])

        # Mask pred
        # Thresholding the output values
        binary_pred_mask = (outputs[n]["masks"] > 0.5).int()
        if binary_pred_mask.shape[0] > 0:
            if shift_by_1:
                remapped_pred_labels = outputs[n]["labels"] - 1
            else:
                remapped_pred_labels = outputs[n]["labels"]
            remapped_pred_labels = remap_labels(remapped_pred_labels, ds.num_classes, ignored_indices)
            padded_binary_pred_mask = insert_component_masks(
                binary_pred_mask,
                remapped_pred_labels,
                ds.num_classes,  # num_classes + 1 if ignored_indices[0] is not None else num_classes,
                ignore_index=ds.ignore_index
            ).to(device)
            if remove_overlap:
                padded_binary_pred_mask = remove_mask_overlap(padded_binary_pred_mask.unsqueeze(0)).squeeze(0)
            if target_size is not None:
                padded_binary_pred_mask = F.interpolate(padded_binary_pred_mask.unsqueeze(0).float(), size=target_size,
                                               mode='nearest').squeeze(0)
            int_pred_mask = convert_to_integer_mask(padded_binary_pred_mask.unsqueeze(0),
                                                    num_classes=ds.num_classes,
                                                    ignore_index=ds.ignore_index)
            rgb_pred_mask = convert_mask_to_RGB(int_pred_mask, palette=ds.get_cmap().to(device), ignore_index=ds.ignore_index)
            rgb_pred_mask = rgb_pred_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            rgb_pred_mask = np.zeros_like(rgb_gt_mask)
        ax[4, n].imshow(rgb_pred_mask)
        ax[4, n].set_xticks([])
        ax[4, n].set_yticks([])

    plt.tight_layout()
    plt.savefig(exp_dir / name, bbox_inches='tight')
    plt.close()


def visualise_loss_and_metrics(train_loss_per_ep: list,
                               val_loss_per_ep: list,
                               f1_per_ep: list,
                               iou_per_ep: list,
                               dice_per_ep: list,
                               epoch: int,
                               val_freq: int,
                               exp_dir: Path):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.arange(0, epoch + 1, val_freq), train_loss_per_ep, label='train')
    ax[0].plot(np.arange(0, epoch + 1, val_freq), val_loss_per_ep, label='val')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()
    ax[1].plot(np.arange(0, epoch + 1, val_freq), f1_per_ep, label='Class F1')
    ax[1].plot(np.arange(0, epoch + 1, val_freq), iou_per_ep, label='BB IoU')
    ax[1].plot(np.arange(0, epoch + 1, val_freq), dice_per_ep, label='Mask Dice')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    plt.savefig(exp_dir / 'loss_per_epoch.png')
    plt.close()


def load_ds(chckpt: Path, data: Path, img_size: tuple) -> (Dataset, list[int], bool):
    if 'CADIS' in str(chckpt).upper():
        ds = CaDISv2_Dataset(
            root=data,
            spatial_transform=A.Compose([
                A.Resize(*img_size)
            ]),
            img_normalization=A.Normalize(0.0, 1.0),
            exp=2,
            mode='test',
            # mode='full',
            filter_mislabeled=True,
            sample_sem_label=False
        )
        ignore_indices = [255]
        keep_ignore = False
    elif 'CHOLECSEG' in str(chckpt).upper():
        ds = CholecSeg8kDataset(
            root=data,
            spatial_transform=A.Compose([
                A.Resize(*img_size)
            ]),
            img_normalization=A.Normalize(0.0, 1.0),
            # mode='test'
            mode='full'
        )
        ignore_indices = [0]
        # ignore_indices = []
        keep_ignore = True
    elif 'CATARACT' in str(chckpt).upper():
        ds = Cataract1kSegmentationDataset(
            root=data,
            spatial_transform=A.Compose([
                A.Resize(*img_size)
            ]),
            img_normalization=A.Normalize(0.0, 1.0),
            mode='test',
            # mode='full',
            sample_sem_label=False
        )
        ignore_indices = [0]
        keep_ignore = True
    else:
        raise ValueError("Could not determine dataset from checkpoint path.")

    return ds, ignore_indices, keep_ignore


def load_maskrcnn_overseer(chckpt: Path,
                           data: Path,
                           device: str = 'cuda') -> (nn.Module, Dataset, int, str, list[int], bool, bool):
    ds, ignore_indices, keep_ignore = load_ds(chckpt, data, img_size=(299, 299))
    if 'CADIS' in str(chckpt).upper():
        hidden_ft = 32  # 16
        backbone = 'ResNet18'  # 'ResNet34'
    elif 'CHOLECSEG' in str(chckpt).upper():
        hidden_ft = 64
        backbone = 'ResNet50'
    elif 'CATARACT' in str(chckpt).upper():
        hidden_ft = 32
        backbone = 'ResNet18'
    else:
        raise ValueError("Could not determine dataset from checkpoint path.")

    m = get_model_instance_segmentation(
        num_classes=ds.num_classes - len(ignore_indices) + 1,
        trainable_backbone_layers=0,
        hidden_ft=hidden_ft,
        custom_in_ft_box=None,
        custom_in_ft_mask=None,
        backbone=backbone,
        img_size=(299, 299)
    )
    m.load_state_dict(torch.load(chckpt, weights_only=True, map_location='cpu'))
    m = m.to(device)
    m.eval()

    return m, ds, hidden_ft, backbone, ignore_indices, True, keep_ignore


def load_DETR_overseer(chckpt: Path,
                       data: Path,
                       device: str = 'cuda') -> (nn.Module, nn.Module, Dataset, int, list[int], bool):

    ds, ignore_indices, keep_ignore = load_ds(chckpt, data, img_size=(200, 200))

    if 'CADIS' in str(chckpt).upper():
        num_queries = 100
    elif 'CHOLECSEG' in str(chckpt).upper():
        num_queries = 100
    elif 'CATARACT' in str(chckpt).upper():
        num_queries = 100
    else:
        raise ValueError("Could not determine dataset from checkpoint path.")

    num_train_classes = ds.num_classes - len(ignore_indices) + 1  # +1 for BG

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    m = DetrForSegmentation.from_pretrained(
        "facebook/detr-resnet-50-panoptic",
        num_labels=num_train_classes,
        ignore_mismatched_sizes=True,
        num_queries=num_queries,
    ).to(device)
    m.load_state_dict(torch.load(chckpt, map_location='cpu', weights_only=True))
    m.eval()

    return m, processor, ds, num_train_classes, ignore_indices, keep_ignore


def load_Mask2Former_overseer(chckpt: Path,
                              data: Path,
                              device: str = "cuda"):

    ds, ignore_indices, keep_ignore = load_ds(chckpt, data, img_size=(299, 299))

    num_train_classes = ds.num_classes - len(ignore_indices) + 1  # +1 for BG

    backbone = 'swin-base-coco-instance'

    m = Mask2FormerForUniversalSegmentation.from_pretrained(
        f"facebook/mask2former-{backbone}",
        num_labels=num_train_classes,  # Adapt the number of classes
        ignore_mismatched_sizes=True,  # To adapt pretrained weights if num_classes changes,
        num_queries=20
    ).to(device)
    m.load_state_dict(torch.load(chckpt, map_location='cpu', weights_only=True))
    m.eval()

    return m, ds, num_train_classes, ignore_indices, keep_ignore


def create_detr_targets(targets: list, img_size: tuple):
    """

    Input targets boxes are in [x_min, y_min, x_max, y_max] with ABSOLUTE pixel vales!

    :param targets: List of targets in the old format
    :param img_size: Tuple of (H, W) image size
    :return: List of targets in DETR format
    """
    detr_targets = []
    for target in targets:

        normalized_boxes = target["boxes"].clone()
        normalized_boxes[:, [0, 2]] /= img_size[1]  # Normalize x_min and x_max
        normalized_boxes[:, [1, 3]] /= img_size[0]  # Normalize y_min and y_max

        detr_target = {
            "boxes": normalized_boxes,
            "class_labels": target["labels"],
            "masks": target["masks"]
        }
        detr_targets.append(detr_target)
    return detr_targets


def process_detr_outputs(outputs, image_size: tuple, num_labels: int, threshold: float = 0.5):
    """
    Post-process DETR outputs to extract bounding boxes, classes, and binary masks.
    """

    batch_size = outputs.logits.shape[0]
    per_sample_results = []

    for n in range(batch_size):

        # Extract per-sample predictions
        logits = outputs.logits[n]  # [num_queries, num_classes]
        boxes = outputs.pred_boxes[n]  # [num_queries, 4], values in [0,1] and in [x_min, y_min, x_max, y_max] form
        masks = outputs.pred_masks[n]  # [num_queries, height, width]

        # Compute class probabilities
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        scores, labels = probs.max(dim=-1)  # Get max probability and corresponding class label

        # Filter "no-object" predictions and low-confidence predictions
        # NOTE: DETR assigns "no-object" to id num_classes + 1
        keep_mask = (labels != num_labels) & (scores > threshold)

        # Binarize masks
        binary_masks = (masks[keep_mask] > 0.5).to(torch.uint8)

        # Denormalize to pixel coordinates
        pixel_boxes = boxes[keep_mask].clone()
        pixel_boxes[:, [0, 2]] *= image_size[1]  # Denormalize x_min and x_max
        pixel_boxes[:, [1, 3]] *= image_size[0]  # Denormalize y_min and y_max

        # Append per-sample results
        per_sample_results.append({
            "scores": scores[keep_mask],
            "labels": labels[keep_mask],
            "boxes": pixel_boxes,
            "masks": binary_masks
        })

    return per_sample_results


def process_mask2former_outputs(outputs, image_size: tuple, num_labels: int, threshold: float = 0.5):
    """
    Processes Mask2Former outputs to extract scores, labels, masks, and bounding boxes.

    Args:
        outputs: Mask2Former model outputs.
        image_size (tuple): Tuple of (height, width) for the original image.
        num_labels (int): Number of foreground classes (including no-object class).
        threshold (float): Confidence threshold for predictions.

    Returns:
        list[dict]: A list of dictionaries, one per sample, containing:
            - "scores": Confidence scores for the kept queries.
            - "labels": Class labels for the kept queries.
            - "boxes": Bounding boxes in (x_min, y_min, x_max, y_max) format (absolute pixel values).
            - "masks": Binary masks for the kept queries.
    """

    batch_size = outputs.class_queries_logits.shape[0]
    per_sample_results = []

    for n in range(batch_size):

        # Extract outputs
        class_logits = outputs.class_queries_logits[n]  # [num_queries, num_labels]
        mask_logits = outputs.masks_queries_logits[n]  # [num_queries, height, width]

        # Apply softmax to class logits
        probs = torch.softmax(class_logits, dim=-1)  # Convert to probabilities
        scores, labels = probs.max(dim=-1)  # Get max score and predicted class for each query

        # Filter "no-object" predictions and low-confidence predictions
        keep_mask = (labels != num_labels) & (scores > threshold)

        # Keep only the masks and scores corresponding to valid queries
        kept_scores = scores[keep_mask]
        kept_labels = labels[keep_mask]
        kept_masks = mask_logits[keep_mask]
        kept_logits = class_logits[keep_mask]

        # Upsample masks to original image size
        upsampled_masks = F.interpolate(
            kept_masks.unsqueeze(1),  # Add channel dimension
            size=image_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # Remove channel dimension

        # Threshold upsampled masks
        binary_masks = (upsampled_masks > threshold).to(torch.uint8)

        # Compute bounding boxes for each binary mask
        pixel_boxes = []
        for mask in binary_masks:
            # Find the non-zero region of the mask
            non_zero_coords = torch.nonzero(mask)
            if non_zero_coords.size(0) == 0:  # Empty mask, skip
                pixel_boxes.append([0, 0, 0, 0])
            else:
                y_min, x_min = non_zero_coords.min(dim=0).values.tolist()
                y_max, x_max = non_zero_coords.max(dim=0).values.tolist()
                pixel_boxes.append([x_min, y_min, x_max, y_max])

        per_sample_results.append({
            "logits": kept_logits,
            "scores": kept_scores,
            "labels": kept_labels,
            "boxes": torch.tensor(pixel_boxes, dtype=torch.float32, device=binary_masks.device),
            "masks": binary_masks,
            "logit_masks": upsampled_masks
        })

    return per_sample_results
