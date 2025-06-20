import os
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import write_png
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

from sds_playground.utils import convert_to_integer_mask, convert_mask_to_RGB

from src.data import remap_labels, insert_component_masks
from src.utils import load_maskrcnn_overseer


def main(frames_root: Path,
         tgt_dir: Path,
         checkpoint: Path,
         data_root: Path,
         vid_name_pattern: str | None = None,
         device: str = 'cuda'):
    assert frames_root.is_dir(), f"{frames_root} not a directory."
    assert checkpoint.exists(), f"{checkpoint} doesnt exist."
    tgt_dir.mkdir(parents=True, exist_ok=True)

    m, ds, _, _, ignore_indices, shift_by_1, _ = load_maskrcnn_overseer(checkpoint, data_root, device=device)

    # Transformation for input images
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((299, 299)),
        T.Normalize(0.0, 1.0)
    ])

    # Get list of video directories
    video_dirs = [d for d in os.listdir(frames_root)
                  if os.path.isdir(os.path.join(frames_root, d))]
    video_dirs = natsorted(video_dirs)

    for video_dir in video_dirs:

        # Skip unwanted videos / sub-folders (annotations etc.)
        if vid_name_pattern is not None:
            if vid_name_pattern not in video_dir:
                continue

        frames_video_dir = frames_root / video_dir
        pred_video_dir = tgt_dir / video_dir
        pred_video_dir.mkdir(parents=True, exist_ok=True)

        # Get sorted list of frame files
        frame_files = natsorted(
            glob(os.path.join(frames_video_dir, '**/', '*.jpg'), recursive=True))  # Adjust extension if needed

        for t in tqdm(range(len(frame_files)), desc=f'Predicting {video_dir}'):

            frame_name = frame_files[t].split("/")[-1].split(".")[0]

            img = Image.open(frame_files[t]).convert('RGB')
            img = transform(img).to(device)

            with torch.no_grad():
                outputs = m([img])
                pred_mask = outputs[0]["masks"]
                pred_labels = outputs[0]["labels"]
                if shift_by_1:
                    pred_labels -= 1
                remapped_pred_labels = remap_labels(pred_labels, ds.num_classes, ignore_indices)
                # print(remapped_pred_labels)
                binary_pred_mask = (pred_mask > 0.5).int()
                padded_binary_pred_mask = insert_component_masks(
                    binary_pred_mask,
                    remapped_pred_labels,
                    ds.num_classes,  # num_classes + 1 if ignored_indices[0] is not None else num_classes,
                    ignore_index=ignore_indices[0] if ignore_indices[0] is not None else None
                ).to(device)
                int_pred_mask = convert_to_integer_mask(
                    padded_binary_pred_mask.unsqueeze(0),
                    num_classes=ds.num_classes,
                    ignore_index=ignore_indices[0] if ignore_indices[0] is not None else None)
                rgb_pred_mask = convert_mask_to_RGB(
                    int_pred_mask, palette=ds.get_cmap().to(device),
                    ignore_index=ignore_indices[0] if ignore_indices[0] is not None else None)
                rgb_pred_mask = rgb_pred_mask.squeeze(0)
                write_png((rgb_pred_mask.cpu() * 255).to(torch.uint8),
                          filename=str(pred_video_dir / f"{frame_name}_rgb_mask.png"))
                # write_png(int_pred_mask.cpu().to(torch.uint8),
                #          filename=str(pred_video_dir / f"{frame_name}_int_mask.png"))
                np.savez_compressed(file=str(pred_video_dir / f"{frame_name}_binary_mask.npz"),
                                    arr=padded_binary_pred_mask.to(torch.uint8).cpu().numpy())

                """
                # Uncomment if not needed (compute intensive)
                num_c = padded_binary_pred_mask.shape[0]
                fig, ax = plt.subplots(nrows=2, ncols=num_c, figsize=(num_c*3, 3))
                ax[0, 0].imshow((rgb_pred_mask * 255).int().permute(1, 2, 0).cpu().numpy())
                ax[0, 0].axis('off')
                for c in range(num_c):
                    ax[0, c].axis('off')
                    ax[1, c].imshow(padded_binary_pred_mask[c].cpu().numpy())
                    ax[1, c].set_title(str(c))
                    ax[1, c].axis('off')
                plt.tight_layout()
                plt.savefig(str(pred_video_dir / f"{frame_name}_per_label_plot.png"), bbox_inches='tight')
                plt.close()
                """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="Dataset root directory.")
    parser.add_argument("--frames_root", type=str, help="Path to frames in .png format.")
    parser.add_argument("--pattern", type=str, help="Video name pattern.", default=None)
    parser.add_argument("--tgt_dir", type=str, help="Path to save predictions.")
    parser.add_argument("--checkpoint", type=str, help="Path to Mask R-CNN checkpoint.")
    parser.add_argument("--device", type=str, help="Device literal for inference.")
    args = parser.parse_args()

    main(Path(args.frames_root),
         Path(args.tgt_dir),
         Path(args.checkpoint),
         Path(args.data_root),
         args.pattern,
         args.device)
