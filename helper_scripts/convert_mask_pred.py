import os.path
from glob import glob
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from natsort import natsorted


def convert_mask_pred(root: str):

    assert os.path.isdir(root)

    mask_files = natsorted(glob(os.path.join(args.root, '*/*.npz'), recursive=True))
    for mask_file in tqdm(mask_files, desc=f'Converting {args.root}'):
        npz_mask = np.load(mask_file, allow_pickle=True)
        npz_mask = npz_mask["arr"].astype(np.uint8)
        mask_tensor = torch.from_numpy(npz_mask).to(torch.uint8).cuda()
        resized_mask = F.interpolate(mask_tensor.unsqueeze(0), size=(299, 299), mode='nearest').squeeze(0)
        np.savez_compressed(file=mask_file, arr=resized_mask.cpu().numpy())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()
    convert_mask_pred(args.root)
