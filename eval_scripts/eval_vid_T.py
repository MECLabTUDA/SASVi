import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from glob import glob
from skimage import measure
from tqdm import tqdm
from natsort import natsorted

from sds_playground.utils import convert_to_integer_mask

from src.utils import remove_mask_overlap


def compute_iou(mask1, mask2):
    # mask1 and mask2 are PyTorch tensors on GPU
    intersection = torch.logical_and(mask1, mask2).sum().item()
    union = torch.logical_or(mask1, mask2).sum().item()
    if union == 0:
        return np.nan  # Label not present in either mask
    iou = intersection / union
    return iou


def compute_contour_distance(mask1, mask2):
    # mask1 and mask2 are PyTorch tensors on GPU
    # Move masks to CPU and convert to numpy arrays
    mask1_cpu = mask1.cpu().numpy().astype(np.uint8)
    mask2_cpu = mask2.cpu().numpy().astype(np.uint8)

    # Find contours for each mask
    contours1 = measure.find_contours(mask1_cpu, level=0.5)
    contours2 = measure.find_contours(mask2_cpu, level=0.5)

    if not contours1 or not contours2:
        return np.nan  # Label not present in either mask

    # Concatenate all contour points
    points1 = np.vstack(contours1)
    points2 = np.vstack(contours2)

    # Compute distances from points1 to points2
    points1_tensor = torch.from_numpy(points1).to(mask1.device)
    points2_tensor = torch.from_numpy(points2).to(mask1.device)

    distances = torch.cdist(points1_tensor, points2_tensor)
    min_distances_1 = torch.min(distances, dim=1)[0]
    min_distances_2 = torch.min(distances, dim=0)[0]

    # Average symmetric contour distance
    avg_distance = (min_distances_1.mean().item() + min_distances_2.mean().item()) / 2.0

    return avg_distance


def main(segmentation_root: str,
         vid_file_pattern: str | None = None,
         mask_file_pattern: str = '*.png',
         ignore_index: int | None = None,
         device: str = 'cuda'):

    # Get list of video directories
    video_dirs = [os.path.join(segmentation_root, d) for d in os.listdir(segmentation_root)
                  if os.path.isdir(os.path.join(segmentation_root, d))]
    video_dirs = natsorted(video_dirs)

    per_label_iou_scores = defaultdict(list)  # label -> list of IoU scores
    per_label_contour_distances = defaultdict(list)  # label -> list of contour distances

    for video_dir in video_dirs:

        # Skip unwanted videos / sub-folders (annotations etc.)
        if vid_file_pattern is not None:
            if vid_file_pattern not in video_dir:
                continue

        # Get sorted list of mask files
        mask_files = natsorted(glob(os.path.join(video_dir, mask_file_pattern)))

        if len(mask_files) < 2:
            raise Exception("Need at least two frames.")

        for t in tqdm(range(len(mask_files) - 1), desc=f'Evaluating {video_dir} vid_T'):

            # Load masks at time t and t+1
            if mask_file_pattern.endswith('.png'):
                mask_t = np.array(Image.open(mask_files[t]))
                mask_t1 = np.array(Image.open(mask_files[t + 1]))
            elif mask_file_pattern.endswith('.npz'):
                mask_t = np.load(mask_files[t])['arr']
                mask_t1 = np.load(mask_files[t + 1])['arr']
            else:
                raise NotImplementedError

            # Convert masks to PyTorch tensors and move to GPU
            mask_t_tensor = torch.from_numpy(mask_t).to(device).to(torch.uint8).unsqueeze(0)
            if mask_t_tensor.dim() < 4:
                mask_t_tensor = mask_t_tensor.unsqueeze(0)
            mask_t1_tensor = torch.from_numpy(mask_t1).to(device).to(torch.uint8).unsqueeze(0)
            if mask_t1_tensor.dim() < 4:
                mask_t1_tensor = mask_t1_tensor.unsqueeze(0)

            if mask_file_pattern.endswith('.npz'):
                mask_t_tensor = remove_mask_overlap(mask_t_tensor)
                mask_t_tensor = convert_to_integer_mask(
                    mask_t_tensor, num_classes=mask_t_tensor.shape[1], ignore_index=ignore_index).unsqueeze(1)
                mask_t1_tensor = remove_mask_overlap(mask_t1_tensor)
                mask_t1_tensor = convert_to_integer_mask(
                    mask_t1_tensor, num_classes=mask_t1_tensor.shape[1], ignore_index=ignore_index).unsqueeze(1)

            mask_t_tensor = F.interpolate(mask_t_tensor.float(), size=(299, 299), mode='nearest').squeeze().to(torch.uint8)
            mask_t1_tensor = F.interpolate(mask_t1_tensor.float(), size=(299, 299), mode='nearest').squeeze().to(torch.uint8)

            # Get all labels present in either mask
            labels_t = torch.unique(mask_t_tensor)
            labels_t1 = torch.unique(mask_t1_tensor)
            labels = torch.unique(torch.cat((labels_t, labels_t1)))

            for label in labels:
                # Uncomment the following lines to skip background label (e.g., label 0)
                # if label.item() == 0:
                #     continue

                # Create binary masks for the label
                mask1 = (mask_t_tensor == label)
                mask2 = (mask_t1_tensor == label)
                # Compute IoU
                iou = compute_iou(mask1, mask2)
                if not np.isnan(iou):
                    per_label_iou_scores[label.item()].append(iou)
                # Compute contour distance
                contour_distance = compute_contour_distance(mask1, mask2)
                if not np.isnan(contour_distance):
                    per_label_contour_distances[label.item()].append(contour_distance)

    # Compute average per-label IoU and contour distances
    avg_iou_per_label = {}
    avg_contour_distance_per_label = {}

    for label in per_label_iou_scores:
        avg_iou_per_label[label] = np.mean(per_label_iou_scores[label])

    for label in per_label_contour_distances:
        avg_contour_distance_per_label[label] = np.mean(per_label_contour_distances[label])

    # Compute macro averages over labels
    macro_avg_iou = np.mean(list(avg_iou_per_label.values()))
    macro_avg_contour_distance = np.mean(list(avg_contour_distance_per_label.values()))

    # Print per-label averages
    print("Per-label IoU scores:")
    for label in sorted(avg_iou_per_label):
        print(f"Label {label}: IoU = {avg_iou_per_label[label]:.4f}")

    print("\nPer-label Contour Distances:")
    for label in sorted(avg_contour_distance_per_label):
        print(f"Label {label}: Contour Distance = {avg_contour_distance_per_label[label]:.4f}")

    # Print macro averages
    print(f"\nMacro-average IoU score over labels: {macro_avg_iou:.4f}")
    print(f"Macro-average Contour Distance over labels: {macro_avg_contour_distance:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute per-label Contour Distance and IoU scores between consecutive segmentations in videos.')
    parser.add_argument('--segm_root', type=str, help='Root directory containing video segmentation directories.')
    parser.add_argument('--vid_pattern', type=str, help='Pattern for loading video directories.', default=None)
    parser.add_argument('--mask_pattern', type=str, help='Pattern for loading mask files.')
    parser.add_argument('--ignore', type=int, help='Ignore index')
    parser.add_argument('--device', type=str, help='Device literal.')
    args = parser.parse_args()
    main(args.segm_root,
         args.vid_pattern,
         args.mask_pattern,
         args.ignore,
         args.device)
