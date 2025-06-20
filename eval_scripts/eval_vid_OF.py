import os
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from PIL import Image
from tqdm import tqdm
from natsort import natsorted

from sds_playground.utils import convert_to_integer_mask

from src.utils import remove_mask_overlap


def compute_dice(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    # mask1 and mask2 are PyTorch tensors on GPU
    intersection = torch.logical_and(mask1, mask2).sum().item()
    size1 = mask1.sum().item()
    size2 = mask2.sum().item()
    if size1 + size2 == 0:
        return np.nan  # Label not present in either mask
    dice = 2 * intersection / (size1 + size2)
    return dice


def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    # mask1 and mask2 are PyTorch tensors on GPU
    intersection = torch.logical_and(mask1, mask2).sum().item()
    union = torch.logical_or(mask1, mask2).sum().item()
    if union == 0:
        return np.nan  # Label not present in either mask
    iou = intersection / union
    return iou


def warp_segmentation(mask: np.ndarray, flow: torch.Tensor) -> np.ndarray:
    h, w = mask.shape
    flow = flow.cpu().numpy()
    flow = flow.transpose(1, 2, 0)  # (H, W, 2)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    coords_x = grid_x + flow[:, :, 0]
    coords_y = grid_y + flow[:, :, 1]
    coords_x = np.clip(coords_x, 0, w - 1)
    coords_y = np.clip(coords_y, 0, h - 1)
    warped_mask = mask[coords_y.astype(np.int32), coords_x.astype(np.int32)]
    return warped_mask


def main(segmentation_root: str,
         frames_root: str,
         vid_file_pattern: str | None = None,
         frame_file_pattern: str = '*.jpg',
         mask_file_pattern: str = '*.png',
         ignore_index: int | None = None,
         raft_iters: int = 12,
         device: str = 'cuda'):

    # Load the RAFT model from torchvision
    model = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=True).to(device)
    model.eval()

    # Transformation for input images
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((512, 384)),
        T.Normalize(0.5, 0.5)
    ])

    # Get list of video directories
    video_dirs = [d for d in os.listdir(frames_root)
                  if os.path.isdir(os.path.join(frames_root, d))]
    video_dirs = natsorted(video_dirs)

    per_label_iou_scores = defaultdict(list)  # label -> list of IoU scores
    per_label_dice_scores = defaultdict(list)  # label -> list of Dice scores

    for video_dir in video_dirs:

        # Skip unwanted videos / sub-folders (annotations etc.)
        if vid_file_pattern is not None:
            if vid_file_pattern not in video_dir:
                continue

        seg_video_dir = os.path.join(segmentation_root, video_dir)
        frames_video_dir = os.path.join(frames_root, video_dir)
        # print(f"{frames_video_dir} isdir: {os.path.isdir(frames_video_dir)}")

        # Get sorted list of mask files and frame files
        mask_files = natsorted(glob(os.path.join(seg_video_dir, mask_file_pattern)))
        frame_files = natsorted(glob(os.path.join(frames_video_dir, frame_file_pattern)))

        if len(mask_files) < 2 or len(frame_files) < 2:
            raise Exception(f"Need at least two frames"
                            f" but got {len(mask_files)} masks and {len(frame_files)} frames.")
        if abs(len(mask_files) - len(frame_files)) > 1:
            raise Exception(f"Need (more or less) equal amount of frames and masks"
                            f" but got {len(mask_files)} masks and {len(frame_files)} frames.")

        for t in tqdm(range(len(mask_files) - 1), desc=f'Evaluating {video_dir} vid_OF'):
            # Load frames at time t and t+1
            frame_t = Image.open(frame_files[t]).convert('RGB')
            frame_t1 = Image.open(frame_files[t + 1]).convert('RGB')

            # Transform frames
            img1 = transform(frame_t).unsqueeze(0).to(device)
            img2 = transform(frame_t1).unsqueeze(0).to(device)

            # The RAFT model expects a list of image pairs
            with torch.no_grad():
                flow_list = model(img1, img2, num_flow_updates=raft_iters)
                flow = flow_list[-1][0]  # Shape: (2, H, W)

            flow = F.interpolate(flow.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)
            # flow = F.interpolate(flow.unsqueeze(0), size=(300, 300), mode='bilinear', align_corners=False).squeeze(0)

            # Load masks at time t and t+1
            if mask_file_pattern.endswith('.png'):
                mask_t = np.array(Image.open(mask_files[t]))
                mask_t1 = np.array(Image.open(mask_files[t + 1]))
            elif mask_file_pattern.endswith('.npz'):
                mask_t = np.load(mask_files[t])['arr']
                mask_t1 = np.load(mask_files[t + 1])['arr']
            else:
                raise NotImplementedError

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
            #mask_t_tensor = F.interpolate(mask_t_tensor.float(), size=(300, 300), mode='nearest').squeeze().to(
            #    torch.uint8)
            #mask_t1_tensor = F.interpolate(mask_t1_tensor.float(), size=(300, 300), mode='nearest').squeeze().to(
            #    torch.uint8)

            mask_t = mask_t_tensor.cpu().numpy()

            # Warp mask_t using the computed flow to get predicted mask at t+1
            warped_mask_t1 = warp_segmentation(mask_t, flow)
            warped_mask_t1_tensor = torch.from_numpy(warped_mask_t1).to(device)

            # Get all labels present in either warped mask or actual mask at t+1
            labels_warped = torch.unique(warped_mask_t1_tensor)
            labels_actual = torch.unique(mask_t1_tensor)
            labels = torch.unique(torch.cat((labels_warped, labels_actual)))

            for label in labels:
                # Skip background label if needed
                # if label.item() == 0:
                #     continue
                # Create binary masks for the label
                warped_label_mask = (warped_mask_t1_tensor == label)
                actual_label_mask = (mask_t1_tensor == label)
                # Compute IoU
                iou = compute_iou(warped_label_mask, actual_label_mask)
                if not np.isnan(iou):
                    per_label_iou_scores[label.item()].append(iou)
                # Compute Dice
                dice = compute_dice(warped_label_mask, actual_label_mask)
                if not np.isnan(dice):
                    per_label_dice_scores[label.item()].append(dice)

    # Compute average per-label IoU and Dice scores
    avg_iou_per_label = {}
    avg_dice_per_label = {}

    for label in per_label_iou_scores:
        avg_iou_per_label[label] = np.mean(per_label_iou_scores[label])

    for label in per_label_dice_scores:
        avg_dice_per_label[label] = np.mean(per_label_dice_scores[label])

    # Compute macro averages over labels
    macro_avg_iou = np.mean(list(avg_iou_per_label.values()))
    macro_avg_dice = np.mean(list(avg_dice_per_label.values()))

    # Print per-label averages
    print("Per-label IoU scores:")
    for label in sorted(avg_iou_per_label):
        print(f"Label {label}: IoU = {avg_iou_per_label[label]:.4f}")

    print("\nPer-label Dice scores:")
    for label in sorted(avg_dice_per_label):
        print(f"Label {label}: Dice = {avg_dice_per_label[label]:.4f}")

    # Print macro averages
    print(f"\nMacro-average IoU score over labels: {macro_avg_iou:.4f}")
    print(f"Macro-average Dice score over labels: {macro_avg_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute per-label Dice and IoU scores between warped and actual segmentations using optical flow.')
    parser.add_argument('--segm_root', type=str, help='Root directory containing segmentation masks.')
    parser.add_argument('--frames_root', type=str, help='Root directory containing video frames.')
    parser.add_argument('--vid_pattern', type=str, help='Pattern for loading video directories.', default=None)
    parser.add_argument('--frame_pattern', type=str, help='Pattern for loading frame files.')
    parser.add_argument('--mask_pattern', type=str, help='Pattern for loading mask files.')
    parser.add_argument('--ignore', type=int, help='Ignore index.', default=None)
    parser.add_argument('--raft_iters', type=int, help='Number of RAFT iterations.', default=12)
    parser.add_argument('--device', type=str, help='Device for flow calculation with RAFT.', default='cuda')
    args = parser.parse_args()
    main(args.segm_root,
         args.frames_root,
         args.vid_pattern,
         args.frame_pattern,
         args.mask_pattern,
         args.ignore,
         args.raft_iters,
         args.device)
