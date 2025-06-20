import os
import argparse

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.io.image import write_png
from natsort import natsorted

from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_float_cmap
from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import get_cataract1k_float_cmap
from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import get_cholecseg8k_float_cmap
from sds_playground.utils import convert_mask_to_RGB


def convert_predictions_to_rgb_masks(predictions_dir: str,
                                     dataset: str,
                                     output_dir: str | None = None,
                                     ignore_index: int | None = None):

    if dataset == 'cadisv2':
        cmap = get_cadis_float_cmap()
        flip_by_1 = True  # CaDISv2 does not have 0=BG...
    elif dataset == 'cataract1k':
        cmap = get_cataract1k_float_cmap()
        flip_by_1 = False
    elif dataset == 'cholecseg8k':
        cmap = get_cholecseg8k_float_cmap()
        flip_by_1 = False
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    # Iterate over the directories in the predictions directory
    for dir in natsorted(os.listdir(predictions_dir)):

        full_dir = os.path.join(predictions_dir, dir)
        if not os.path.isdir(full_dir):
            continue

        # Create the output directory if any is given
        if output_dir is not None:
            output_path = os.path.join(output_dir, dir)
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = full_dir

        for mask in tqdm(natsorted(os.listdir(full_dir)), desc=f"Processing directory '{dir}'"):
            mask_path = os.path.join(full_dir, mask)
            output_rgb_mask_path = os.path.join(output_path, mask.replace('.png', '_rgb_mask.png'))

            mask = np.array(Image.open(mask_path))
            integer_mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(torch.int64)

            if flip_by_1:

                integer_mask_tensor -= 1
                integer_mask_tensor[integer_mask_tensor == -1] = ignore_index

            rgb_mask_tensor = convert_mask_to_RGB(integer_mask_tensor, cmap, ignore_index=ignore_index)
            rgb_mask_tensor = (rgb_mask_tensor * 255).to(torch.uint8).squeeze(0)

            write_png(rgb_mask_tensor, output_rgb_mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert nnUNet predictions to RGB mask images')
    parser.add_argument('--predictions', type=str, help='Path to the directory containing the nnUNet predictions.')
    parser.add_argument('--output', type=str, help='Path to the directory where the RGB masks will be saved (default is input dir).', default=None)
    parser.add_argument('--dataset', type=str, help='Name of the dataset.')
    parser.add_argument('--ignore', type=int, help='Ignore index (default None).', default=None)
    args = parser.parse_args()

    convert_predictions_to_rgb_masks(args.predictions, args.dataset, args.output, args.ignore)


