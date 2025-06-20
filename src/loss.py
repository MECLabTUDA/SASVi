import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align
from scipy.optimize import linear_sum_assignment


def loss_dict_str(loss_dict: dict) -> str:
    return (f"classifier: {loss_dict['loss_classifier']}\t"
            f"box_reg: {loss_dict['loss_box_reg']}\t"
            f"mask: {loss_dict['loss_mask']}\t"
            f"objectness: {loss_dict['loss_objectness']}\t"
            f"rpn_box_reg: {loss_dict['loss_rpn_box_reg']}")


def compute_classification_metrics(targets: list[dict], outputs: list[dict], iou_threshold: float = 0.5) -> dict:
    """
    Computes class-wise precision, recall, and F1 scores for Mask R-CNN outputs.

    Parameters:
    - targets (list of dict): Ground truth annotations for each image.
    - outputs (list of dict): Model predictions for each image.
    - iou_threshold (float): IoU threshold to consider a detection as true positive.

    Returns:
    - results (dict): A dictionary containing precision, recall, and F1 scores for each class.
    """
    # Collect all unique classes present in the dataset
    all_classes = set()
    for target, output in zip(targets, outputs):
        all_classes.update(target['labels'].unique().tolist())
        all_classes.update(output['labels'].unique().tolist())
    all_classes = sorted(list(all_classes))

    # Initialize counts for true positives (TP), false positives (FP), and false negatives (FN)
    metrics = {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in all_classes}

    # Process each image in the batch
    for target, output in zip(targets, outputs):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        pred_boxes = output['boxes']
        pred_labels = output['labels']
        pred_scores = output['scores']

        # Process each class separately
        for cls in all_classes:
            # Ground truth boxes and labels for the current class
            gt_mask = gt_labels == cls
            gt_boxes_cls = gt_boxes[gt_mask]

            # Predicted boxes and labels for the current class
            pred_mask = pred_labels == cls
            pred_boxes_cls = pred_boxes[pred_mask]

            # If no ground truths and no predictions, skip this class
            if len(gt_boxes_cls) == 0 and len(pred_boxes_cls) == 0:
                continue

            # If there are no predictions, all ground truths are false negatives
            if len(pred_boxes_cls) == 0:
                metrics[cls]['FN'] += len(gt_boxes_cls)
                continue

            # If there are no ground truths, all predictions are false positives
            if len(gt_boxes_cls) == 0:
                metrics[cls]['FP'] += len(pred_boxes_cls)
                continue

            # Compute the IoU matrix between ground truth and predicted boxes
            iou_matrix = box_iou(gt_boxes_cls, pred_boxes_cls)  # Shape: [num_gt, num_pred]

            # Convert IoU matrix to cost matrix for assignment
            cost_matrix = 1 - iou_matrix.cpu().numpy()

            # Perform optimal matching using the Hungarian algorithm
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

            # Initialize lists to keep track of matched indices
            matched_gt = []
            matched_pred = []

            # Evaluate matches based on IoU threshold
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
                    metrics[cls]['TP'] += 1
                    matched_gt.append(gt_idx)
                    matched_pred.append(pred_idx)
                else:
                    # No match if IoU is below the threshold
                    pass

            # Calculate false negatives and false positives
            num_fn = len(gt_boxes_cls) - len(matched_gt)
            num_fp = len(pred_boxes_cls) - len(matched_pred)
            metrics[cls]['FN'] += num_fn
            metrics[cls]['FP'] += num_fp

    # Compute precision, recall, and F1 score for each class
    results = {}
    for cls in all_classes:
        TP = metrics[cls]['TP']
        FP = metrics[cls]['FP']
        FN = metrics[cls]['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cls] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    return results


def compute_macro_averages(metrics: dict) -> dict:
    precision_list = []
    recall_list = []
    f1_score_list = []

    for cls, metric in metrics.items():
        # Skip the 'macro' key if it's already in metrics
        if cls == 'macro':
            continue
        precision_list.append(metric['precision'])
        recall_list.append(metric['recall'])
        f1_score_list.append(metric['f1_score'])

    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    macro_f1_score = sum(f1_score_list) / len(f1_score_list) if f1_score_list else 0.0

    macro_metrics = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score
    }

    return macro_metrics


def compute_class_iou(targets: list[dict], outputs: list[dict], iou_threshold: float = 0.5) -> dict:
    """
    Computes the average IoU for each class based on the matched predictions and ground truths.

    Parameters:
    - targets (list of dict): Ground truth annotations for each image.
    - outputs (list of dict): Model predictions for each image.
    - iou_threshold (float): IoU threshold to consider a detection as true positive.

    Returns:
    - results (dict): A dictionary containing the average IoU for each class.
    """
    # Collect all unique classes present in the dataset
    all_classes = set()
    for target, output in zip(targets, outputs):
        all_classes.update(target['labels'].unique().tolist())
        all_classes.update(output['labels'].unique().tolist())
    all_classes = sorted(list(all_classes))

    # Initialize lists to accumulate IoU values for each class
    iou_values = {cls: [] for cls in all_classes}

    # Process each image in the batch
    for target, output in zip(targets, outputs):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        pred_boxes = output['boxes']
        pred_labels = output['labels']
        pred_scores = output['scores']

        # Process each class separately
        for cls in all_classes:
            # Ground truth boxes and labels for the current class
            gt_mask = gt_labels == cls
            gt_boxes_cls = gt_boxes[gt_mask]

            # Predicted boxes and labels for the current class
            pred_mask = pred_labels == cls
            pred_boxes_cls = pred_boxes[pred_mask]
            pred_scores_cls = pred_scores[pred_mask]

            # If there are no ground truths or predictions for this class in this image, skip
            if len(gt_boxes_cls) == 0 or len(pred_boxes_cls) == 0:
                continue

            # Compute the IoU matrix between ground truth and predicted boxes
            iou_matrix = box_iou(gt_boxes_cls, pred_boxes_cls)  # Shape: [num_gt, num_pred]

            # Convert IoU matrix to cost matrix for assignment
            cost_matrix = 1 - iou_matrix.cpu().numpy()

            # Perform optimal matching using the Hungarian algorithm
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

            # Evaluate matches based on IoU threshold and accumulate IoUs
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                iou = iou_matrix[gt_idx, pred_idx].item()
                if iou >= iou_threshold:
                    iou_values[cls].append(iou)
                # Else, do not consider this pair as it doesn't meet the threshold

    # Compute average IoU for each class
    results = {}
    for cls in all_classes:
        if len(iou_values[cls]) > 0:
            average_iou = sum(iou_values[cls]) / len(iou_values[cls])
        else:
            average_iou = 0.0  # No matches for this class
        results[cls] = {'average_iou': average_iou}

    return results


def compute_macro_iou(iou_metrics: dict) -> dict:
    iou_list = []

    for cls, metric in iou_metrics.items():
        # Skip the 'macro' key if it's already in metrics
        if cls == 'macro':
            continue
        iou_list.append(metric['average_iou'])

    macro_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0

    return {'macro_average_iou': macro_iou}


def compute_mask_dice_score(targets: list[dict], outputs: list[dict], iou_threshold: float = 0.5):
    """
    Computes the Dice score for segmentation masks for each class based on the matched predictions and ground truths.

    Parameters:
    - targets (list of dict): Ground truth annotations for each image.
    - outputs (list of dict): Model predictions for each image.
    - iou_threshold (float): IoU threshold to consider a detection as true positive.

    Returns:
    - results (dict): A dictionary containing the average Dice score for each class.
    """
    # Collect all unique classes present in the dataset
    all_classes = set()
    for target, output in zip(targets, outputs):
        all_classes.update(target['labels'].unique().tolist())
        all_classes.update(output['labels'].unique().tolist())
    all_classes = sorted(list(all_classes))

    # Initialize lists to accumulate Dice scores for each class
    dice_scores = {cls: [] for cls in all_classes}

    # Process each image in the batch
    for target, output in zip(targets, outputs):
        gt_masks = target['masks']  # Ground truth masks (shape: [num_gt_obj, H, W])
        gt_labels = target['labels']
        pred_masks = output['masks']  # Predicted masks (shape: [num_pred_obj, 1, H, W] or [num_pred_obj, H, W])
        pred_labels = output['labels']
        pred_scores = output['scores']

        # Ensure pred_masks is in [num_pred_obj, H, W]
        if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
            pred_masks = pred_masks[:, 0]

        # Process each class separately
        for cls in all_classes:

            # Ground truth masks and labels for the current class
            gt_mask = gt_labels == cls
            gt_masks_cls = gt_masks[gt_mask]  # Shape: [num_gt_cls, H, W]

            # Predicted masks and labels for the current class
            pred_mask = pred_labels == cls
            pred_masks_cls = pred_masks[pred_mask]  # Shape: [num_pred_cls, H, W]

            # If there are no ground truths or predictions for this class in this image, skip
            if len(gt_masks_cls) == 0 or len(pred_masks_cls) == 0:
                continue

            # Flatten masks for IoU computation
            gt_masks_flat = gt_masks_cls.view(gt_masks_cls.size(0), -1).float()  # Shape: [num_gt_cls, H*W]
            pred_masks_flat = pred_masks_cls.view(pred_masks_cls.size(0), -1).float()  # Shape: [num_pred_cls, H*W]

            # Compute the IoU matrix between ground truth and predicted masks
            intersection = torch.matmul(gt_masks_flat, pred_masks_flat.t())  # Shape: [num_gt_cls, num_pred_cls]
            area_gt = gt_masks_flat.sum(dim=1, keepdim=True)  # Shape: [num_gt_cls, 1]
            area_pred = pred_masks_flat.sum(dim=1, keepdim=True)  # Shape: [num_pred_cls, 1]

            union = area_gt + area_pred.t() - intersection
            iou_matrix = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero

            # Convert IoU matrix to cost matrix for assignment
            cost_matrix = 1 - iou_matrix.cpu().numpy()

            # Perform optimal matching using the Hungarian algorithm
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

            # Evaluate matches based on IoU threshold and accumulate Dice scores
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                iou = iou_matrix[gt_idx, pred_idx].item()
                if iou >= iou_threshold:
                    # Compute Dice score for the matched masks
                    gt_mask_flat = gt_masks_flat[gt_idx]
                    pred_mask_flat = pred_masks_flat[pred_idx]
                    dice_numerator = 2 * (gt_mask_flat * pred_mask_flat).sum()
                    dice_denominator = gt_mask_flat.sum() + pred_mask_flat.sum()
                    dice_score = dice_numerator / (dice_denominator + 1e-6)
                    dice_scores[cls].append(dice_score.item())
                # Else, do not consider this pair as it doesn't meet the threshold

    # Compute average Dice score for each class
    results = {}
    for cls in all_classes:
        if len(dice_scores[cls]) > 0:
            average_dice = sum(dice_scores[cls]) / len(dice_scores[cls])
        else:
            average_dice = 0.0  # No matches for this class
        results[cls] = {'average_dice': average_dice}

    return results


def compute_macro_dice(dice_metrics: dict) -> dict:
    dice_list = []

    for cls, metric in dice_metrics.items():
        # Skip the 'macro' key if it's already in metrics
        if cls == 'macro':
            continue
        dice_list.append(metric['average_dice'])

    macro_dice = sum(dice_list) / len(dice_list) if dice_list else 0.0

    return {'macro_average_dice': macro_dice}


def match_predictions_to_gt(pred_boxes: torch.Tensor,
                            gt_boxes: torch.Tensor,
                            iou_threshold: float = 0.5) -> list:
    """
    Matches predicted boxes to ground truth boxes based on IoU.

    :param pred_boxes: Predicted bounding boxes (tensor of shape [num_objects_pred, 4])
    :param gt_boxes: Ground truth bounding boxes (tensor of shape [num_objects_gt, 4])
    :param iou_threshold: IoU threshold to determine a match
    :return: List of tuples (pred_idx, gt_idx), where pred_idx and gt_idx are indices of matching boxes
    """
    matches = []
    iou_matrix = box_iou(pred_boxes, gt_boxes)

    # Iterate over ground truth objects
    for gt_idx in range(iou_matrix.size(1)):
        # IoU scores for current GT object with all predicted objects
        iou_scores = iou_matrix[:, gt_idx]
        # Find pred. object with highest IoU
        pred_idx = iou_scores.argmax()

        if iou_scores[pred_idx] >= iou_threshold:
            matches.append((pred_idx.item(), gt_idx))

    return matches


def dice_coefficient_with_matching(pred_masks: torch.Tensor,
                                   gt_masks: torch.Tensor,
                                   pred_boxes: torch.Tensor,
                                   gt_boxes: torch.Tensor,
                                   iou_threshold: float = 0.5,
                                   threshold: float = 0.5) -> torch.Tensor:
    """
    Computes the Dice coefficient between predicted and ground truth masks after matching predictions to GT.

    :param pred_masks: Predicted masks (binary, shape [num_objects_pred, H, W])
    :param gt_masks: Ground truth masks (binary, shape [num_objects_gt, H, W])
    :param pred_boxes: Predicted bounding boxes (tensor of shape [num_objects_pred, 4])
    :param gt_boxes: Ground truth bounding boxes (tensor of shape [num_objects_gt, 4])
    :param iou_threshold: IoU threshold to consider a match between a predicted and ground truth object
    :param threshold: Threshold to convert predicted masks to binary
    :return: Dice coefficient
    """

    pred_masks = (pred_masks > threshold).int()

    # Match predicted and GT masks using IoU of bounding boxes
    matches = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=iou_threshold)

    if len(matches) == 0:
        # No matches, return a Dice score of 0
        return torch.tensor(0.0)

    # Dice for matched objects
    dice_scores = []
    for pred_idx, gt_idx in matches:
        intersection = (pred_masks[pred_idx] & gt_masks[gt_idx]).float().sum()
        union = pred_masks[pred_idx].float().sum() + gt_masks[gt_idx].float().sum()
        dice = (2 * intersection) / (union + 1e-6)
        dice_scores.append(dice)

    # Average Dice over all matched objects
    return torch.mean(torch.tensor(dice_scores))


def project_masks_on_boxes(gt_masks: torch.Tensor,
                           boxes: torch.Tensor,
                           matched_idxs: torch.Tensor,
                           M: int) -> torch.Tensor:
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]


def weighted_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, weight):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])
        weight (Tensor)

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
        mask_targets,
        weight
    )
    return mask_loss


def hungarian_matching(pred_class_logits, pred_masks, gt_labels, gt_masks):
    """
    Perform Hungarian matching between predicted queries and ground truth objects.

    Args:
        pred_class_logits (torch.Tensor): [num_queries, num_classes], class logits.
        pred_masks (torch.Tensor): [num_queries, height, width], predicted masks.
        gt_labels (torch.Tensor): [num_objects], ground truth class labels.
        gt_masks (torch.Tensor): [num_objects, height, width], ground truth masks.

    Returns:
        List[Tuple[int, int]]: List of matched indices (prediction_idx, ground_truth_idx).
    """
    num_queries = pred_class_logits.shape[0]
    num_objects = gt_labels.shape[0]

    if num_objects == 0:
        # No ground truth objects; return empty matches
        return []

    # Classification cost (negative log-likelihood of the correct class)
    pred_probs = torch.softmax(pred_class_logits, dim=-1)  # [num_queries, num_classes]
    classification_cost = -pred_probs[:, gt_labels]  # [num_queries, num_objects]

    # Mask cost (Dice loss or binary cross-entropy loss)
    pred_masks = pred_masks.flatten(1)  # [num_queries, height * width]
    gt_masks = gt_masks.flatten(1)  # [num_objects, height * width]

    # Dice loss cost
    intersection = (pred_masks.unsqueeze(1) * gt_masks.unsqueeze(0)).sum(-1)  # [num_queries, num_objects]
    union = pred_masks.sum(-1, keepdim=True) + gt_masks.sum(-1, keepdim=True).T
    dice_cost = 1 - (2 * intersection / (union + 1e-6))  # [num_queries, num_objects]

    # Combine costs
    total_cost = classification_cost + dice_cost

    # Solve Hungarian matching
    pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().detach().numpy())
    return list(zip(pred_indices, gt_indices))


def mask2former_loss(pred_class_logits,
                     pred_masks,
                     gt_labels,
                     gt_masks,
                     matches,
                     classification_loss_fn,
                     mask_loss_fn):
    """
    Compute classification and mask losses based on Hungarian matching.

    Args:
        pred_class_logits (torch.Tensor): [num_queries, num_classes], class logits.
        pred_masks (torch.Tensor): [num_queries, height, width], predicted masks.
        gt_labels (torch.Tensor): [num_objects], ground truth class labels.
        gt_masks (torch.Tensor): [num_objects, height, width], ground truth masks.
        matches (List[Tuple[int, int]]): List of matched indices (prediction_idx, ground_truth_idx).
        classification_loss_fn:
        mask_loss_fn:

    Returns:
        torch.Tensor: Total loss.
    """
    if len(matches) == 0:
        # No matches, no loss to compute
        return torch.tensor(0.0, requires_grad=True, device=pred_class_logits.device)

    matched_pred_indices, matched_gt_indices = zip(*matches)

    # Matched predictions and ground truth
    matched_pred_class_logits = pred_class_logits[list(matched_pred_indices)]
    matched_pred_masks = pred_masks[list(matched_pred_indices)]
    matched_gt_labels = gt_labels[list(matched_gt_indices)]
    matched_gt_masks = gt_masks[list(matched_gt_indices)]

    # Ensure gt_masks is float for BCEWithLogitsLoss
    matched_gt_masks = matched_gt_masks.float()

    # Compute classification loss
    classification_loss = classification_loss_fn(matched_pred_class_logits, matched_gt_labels)

    # Compute mask loss
    mask_loss = mask_loss_fn(matched_pred_masks, matched_gt_masks)

    return classification_loss + mask_loss

