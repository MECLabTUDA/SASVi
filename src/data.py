import os.path
from collections import defaultdict
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

import torch
import numpy as np
from scipy.ndimage import label
from torch.utils.data import Dataset
from tqdm import tqdm

from sds_playground.utils.utils import convert_to_binary_mask


def get_bb_from_mask(binary_mask: torch.Tensor,
                     components: bool = True,
                     min_comp_fraction: float = 0.0) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """ Converts segmentation masks into bounding boxes, labels, and individual object masks by detecting connected components.

    TODO: Use int64 for the masks?

    :param binary_mask: Binary segmentation mask of shape (N, K, H, W), where:
                        N = batch size,
                        K = number of classes (maximum),
                        H, W = height and width of the image.
    :param components: Bool; Use connected component analysis for individual objects
    :param min_comp_fraction: Float; Minimum fraction of image size for considered connected components

    :return: Tuple of:
             - List of bounding boxes for each image in the minibatch, where each bounding box is in (num_objects, 4) format (xmin, ymin, xmax, ymax).
             - List of labels for each image in the minibatch, where each label is in (num_objects,) format corresponding to class indices.
             - List of masks for each image in the minibatch, where each mask is in (num_objects, H, W) format (binary masks for each object).
    """

    N, K, H, W = binary_mask.shape  # N = batch size, K = number of classes
    all_boxes = []
    all_labels = []
    all_masks = []

    for n in range(N):  # Iterate over each image in the batch
        boxes = []
        labels = []
        masks = []

        for k in range(K):  # Iterate over each class in the mask
            # Get the binary mask for the class
            class_mask = binary_mask[n, k].cpu().numpy()  # Convert to numpy array for connected component labeling

            if components:
                # Find connected components in the binary mask for this class
                labeled_mask, num_components = label(class_mask)

                for component in range(1, num_components + 1):  # Iterate over each component
                    pos = np.where(labeled_mask == component)  # Get the positions of the current component

                    if len(pos[0]) > 0:  # If there are pixels in this component
                        ymin = np.min(pos[0])
                        ymax = np.max(pos[0])
                        xmin = np.min(pos[1])
                        xmax = np.max(pos[1])

                        if (ymax - ymin > min_comp_fraction * H or xmax - xmin > min_comp_fraction * W) and \
                            (ymax - ymin > 0 and xmax - xmin > 0):
                            # Create a mask for this individual object (binary mask)
                            component_mask = (labeled_mask == component).astype(np.uint8)

                            # Append the bounding box and corresponding class label (k)
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(k)
                            masks.append(torch.tensor(component_mask, dtype=torch.uint8))
            else:
                class_mask = binary_mask[n, k].cpu().numpy()
                pos = np.where(class_mask == 1)
                if len(pos[0] > 0):
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    if ymax - ymin > 0 and xmax - xmin > 0:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(k)
                        masks.append(torch.tensor(class_mask, dtype=torch.uint8))

        # Convert lists to tensors for each image
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks, dim=0)  # Stack all masks for this image
        else:
            # No objects detected, return empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)

        all_boxes.append(boxes)
        all_labels.append(labels)
        all_masks.append(masks)

        # Return a list of boxes and labels for each image in the batch
    return all_boxes, all_labels, all_masks


def extract_active_masks(gt_masks):
    """ Extracts the active masks (i.e., non-empty and individual connected components) from the ground truth masks.

    :param gt_masks: Ground truth masks in (K, H, W) format, where K is the maximum number of possible classes.
    :return: Active ground truth masks in (num_objects, H, W) format, where num_objects is the number of classes that actually have objects.
    """
    K, H, W = gt_masks.shape  # K = number of classes, H = height, W = width
    active_masks = []

    for k in range(K):  # Iterate over each class
        class_mask = gt_masks[k].cpu().numpy()  # Convert to numpy array for connected component analysis

        # Find connected components in the binary mask for this class
        labeled_mask, num_components = label(class_mask)

        for component in range(1, num_components + 1):  # Iterate over each component
            component_mask = labeled_mask == component  # Binary mask for this component
            if component_mask.sum() > 0:  # Ensure it's not an empty component
                active_masks.append(torch.tensor(component_mask, dtype=torch.uint8))

    if len(active_masks) == 0:
        # If there are no active masks, return a dummy mask of shape (1, H, W)
        return torch.zeros((1, H, W), dtype=torch.uint8)

    # Stack the active masks to get the shape (num_objects, H, W)
    return torch.stack(active_masks, dim=0)


def insert_component_masks(obj_masks, obj_labels, num_ds_classes, ignore_index: int | None = None):
    """ Inserts predicted masks back into a tensor of shape (K, H, W), where K is the number of classes.

    :param obj_masks: Individual object masks (tensor of shape [num_objects, H, W])
    :param obj_labels: Class labels for each object mask (tensor of shape [num_objects])
    :param num_ds_classes: Total number of possible classes (K) of a dataset, including ignore index
    :param ignore_index: Ignore index of the dataset
    :return: A tensor of shape (K, H, W) where each channel corresponds to a class and contains the combined predicted masks.
    """
    # Empty tensor of shape (K, H, W) to store the masks for each class
    output_masks = torch.zeros((num_ds_classes, *obj_masks.shape[-2:]), dtype=obj_masks.dtype).to(obj_masks.device)

    # Loop through each predicted mask and combine it into the corresponding class channel
    for i, label in enumerate(obj_labels):
        # Use a logical OR to combine masks for the same class
        output_masks[label] = output_masks[label] | obj_masks[i]

    if ignore_index is not None and ignore_index >= num_ds_classes:
        non_positive_mask = torch.sum(output_masks[:-1], dim=0) == 0
        output_masks[-1][non_positive_mask] = 1
    elif ignore_index is not None and ignore_index < num_ds_classes:
        non_positive_mask = torch.sum(output_masks, dim=0) == 0
        output_masks[ignore_index][non_positive_mask] = 1

    return output_masks


def remove_mask_overlap(masks: torch.Tensor) -> torch.Tensor:
    """ Removes overlapping masks in binary mask tensors, favoring higher indexed classes.
        (E.g. for visualisation purposes)

    :param masks: (torch.Tensor) Binary masks of shape (N, K, H, W)
    :return: (torch.Tensor) Non-overlapping masks of shape (N, K, H, W)
    """

    N, K, H, W = masks.shape

    output_masks = torch.zeros_like(masks)

    for class_index in range(K - 1, -1, -1):

        current_class_mask = masks[:, class_index] == 1

        # For the current class, set it in the output if there's no overlap
        output_masks[current_class_mask & (output_masks.sum(dim=1) == 0), class_index] = 1

    return output_masks


def prepare_data(images: torch.Tensor,
                 masks: torch.Tensor,
                 ds: Dataset,
                 ignore_indices: List[int],
                 shift_by_1: bool = True,
                 components: bool = True,
                 keep_ignore: bool = True,
                 min_comp_fraction: float = 0.0,
                 device: str = 'cuda'):

    """ Prepare image and mask data for Mask R-CNN training.

    TODO: High CPU load...

    :param images: (torch.Tensor) Batch of training images in (N, 3, H, W)
    :param masks: (torch.Tensor) Batch of integer training masks in (N, H, W)
    :param ds: (Dataset) The torch dataset
    :param ignore_indices: (list[int]) List of ignore/class indices to ignore during training.
            The first index is assumed to be the datasets ignore index.
    :param shift_by_1: Shifting label ids by 1, since Mask R-CNN assumes that 0 is the background
    :param components: Use connected component analysis to get multiple instances of spatially seperated classes
    :param min_comp_fraction: Minimum H/W for components to be considered
    :param device: Device literal
    :return: List of training images, target dictionary in Mask R-CNN format
    """

    binary_masks = convert_to_binary_mask(masks,
                                          num_classes=ds.num_classes,
                                          ignore_index=ds.ignore_index,
                                          keep_ignore_index=keep_ignore)

    valid_ids = []
    for id in range(0, ds.num_classes - 1 \
            if (ds.ignore_index is not None and ds.ignore_index >= ds.num_classes) else ds.num_classes):
        if id not in ignore_indices:
            valid_ids.append(id)
    reduced_binary_masks = binary_masks[:, [*valid_ids], ...]

    bbs_list, labels_list, masks_list = get_bb_from_mask(reduced_binary_masks, components, min_comp_fraction)

    del binary_masks, reduced_binary_masks

    # Shifting label ids by 1, since Mask R-CNN assumes that 0 is the background
    if shift_by_1:
        labels_list = [label + 1 for label in labels_list]

    images_list = list(image.to(device) for image in images)

    # Create the target dictionaries for each image
    targets = []
    for i in range(len(images_list)):
        target = {
            "boxes": bbs_list[i].to(device),
            "masks": masks_list[i].to(device),
            "labels": labels_list[i].to(device)
        }
        targets.append(target)

    return images_list, targets


def compute_class_weights(dataset: Dataset, num_classes: int, alpha: float = 0.2) -> torch.Tensor:
    """
    Compute class weights (for weighted losses) based on the inverse of their frequency in the dataset.

    :param dataset: (torch.utils.data.Dataset) The dataset with images and segmentation masks.
    :param num_classes: (int) The number of classes in the segmentation task.
    :param alpha: (float) Power transformation factor between 0 and 1 reduce drastic weighting

    :return: (torch.Tensor) Class weights inversely correlated with class frequency.
    """

    if os.path.isfile(f'{type(dataset).__name__}_class_weights.pth'):
        return torch.load(f'{type(dataset).__name__}_class_weights.pth', weights_only=True)

    class_frequencies = defaultdict(int)

    # Iterate over the dataset to collect class frequencies
    for _, mask, _, _ in tqdm(dataset, desc='Computing class weights'):  # Assuming the dataset returns (image, mask) pairs
        mask = np.array(mask)
        for class_idx in range(num_classes):
            class_frequencies[class_idx] += np.sum(mask == class_idx)

    # Total number of pixels in the dataset
    total_pixels = sum(class_frequencies.values())

    # Compute class weights as inverse of class frequency
    class_weights = []
    for class_idx in range(num_classes):
        class_frequency = class_frequencies[class_idx] / total_pixels
        class_weight = (1.0 / (class_frequency + 1e-6)) ** alpha  # Add small value to avoid division by zero
        class_weights.append(class_weight)

    # Normalize the weights so that their sum equals the number of classes
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.sum() * num_classes

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    torch.save(class_weights, f'{type(dataset).__name__}_class_weights.pth')

    return class_weights


def compute_sample_weights(dataset: Dataset, num_classes: int):

    """ Computes weights for weighted sampling based on the average class frequency
        of each class present in the image.

    :param dataset: The pytorch dataset from which to compute the weights
    :param num_classes:  The number of classes in that dataset
    :return:
    """

    if os.path.isfile(f'{type(dataset).__name__}_sample_weights.npy'):
        return np.load(f'{type(dataset).__name__}_sample_weights.npy')

    class_weights = compute_class_weights(dataset, num_classes)

    # Ensure class_weights is a tensor
    if not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    sample_weights = []

    # Iterate over the dataset to compute the sample weights
    for _, mask, _, _ in tqdm(dataset, desc='Computing sampling weights'):
        mask = np.array(mask)  # Convert mask to a NumPy array if necessary

        # Get the unique classes in the mask and their corresponding weights
        unique_classes = np.unique(mask)  # Unique class indices in the mask
        unique_classes = unique_classes[unique_classes < num_classes]  # Remove any invalid classes

        # Average the weights of the unique classes in the mask
        if len(unique_classes) > 0:
            sample_weight = class_weights[unique_classes].mean().item()
        else:
            # In case the mask doesn't contain any valid classes, assign a minimal weight
            sample_weight = 0.0

        sample_weights.append(sample_weight)

    sample_weights = np.array(sample_weights)
    np.save(f'{type(dataset).__name__}_sample_weights.npy', sample_weights)

    return sample_weights


def remap_labels(reduced_labels: torch.Tensor,
                 num_classes: int,
                 ignored_indices: list) -> torch.Tensor:
    """ Re-maps tensor of reduced labels in (K, 1) to original set of labels in (num_classes, 1) """

    original_classes = torch.tensor([*range(num_classes)])

    if len(ignored_indices) > 0:
        if ignored_indices[0] is None or ignored_indices[0] > num_classes:
            ignored_classes = torch.tensor([*ignored_indices[1:]])
        else:
            ignored_classes = torch.tensor([*ignored_indices])
    else:
        ignored_classes = torch.tensor([])

    valid_classes = torch.tensor([cls for cls in original_classes if cls not in ignored_classes],
                                 device=reduced_labels.device)


    mapped_labels = valid_classes[reduced_labels]

    return mapped_labels
