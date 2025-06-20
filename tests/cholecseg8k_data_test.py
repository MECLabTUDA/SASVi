import unittest
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader

from sds_playground.datasets import CholecSeg8kDataset
from sds_playground.utils import *

from src.data import prepare_data, insert_component_masks, get_bb_from_mask
from src.utils import visualise_results
from src.loss import dice_coefficient_with_matching, box_iou


class CholecsegDataTest(unittest.TestCase):

    def setUp(self):
        self.size = (299, 299)
        self.ds = CholecSeg8kDataset(
            root=Path('/local/scratch/CholecSeg8k/'),
            mode='full',
            spatial_transform=A.Compose([
                A.Resize(*self.size),
                A.Rotate(limit=(-45, 45), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), mask_value=0, p=0.5)
            ]),
            img_normalization=A.Normalize(0.0, 1.0)
        )

    def testShape(self):
        img, mask, _, _ = self.ds[0]
        self.assertEqual(img.shape, (3, *self.size))
        self.assertEqual(mask.shape, self.size)

    def testConvert(self):
        dl = DataLoader(self.ds, batch_size=8, shuffle=True, num_workers=0)
        img, mask, _, _ = next(iter(dl))

        ignore_ids = [0]
        print(f"Ignoring ids: {ignore_ids}")

        images_list, targets = prepare_data(img, mask, self.ds, ignore_ids,
                                            device='cpu', shift_by_1=True, components=False)
        for n in range(8):
            print(targets[n]["labels"])

        visualise_results(
            images_list=images_list,
            outputs=targets,
            targets=targets,
            ds=self.ds,
            ignored_indices=ignore_ids,
            shift_by_1=True,
            img_norm=(0.0, 1.0),
            device='cpu',
            exp_dir=Path('./'),
            name='cholecseg8k_test.png'
        )

    def testLoss(self):
        dl = DataLoader(self.ds, batch_size=8, shuffle=True, num_workers=0)
        img, mask, _, _ = next(iter(dl))

        ignore_ids = [self.ds.ignore_index, 0, 1, 2, 3]
        print(f"Ignoring ids: {ignore_ids}")

        images_list, targets = prepare_data(img, mask, self.ds, ignore_ids,
                                            device='cpu', shift_by_1=True, components=False)

        for n in range(len(targets)):
            iou_matrix = box_iou(targets[n]["boxes"], targets[n]["boxes"])
            iou_scores = torch.diag(iou_matrix)
            print(f"IoU: {torch.mean(iou_scores).item()}")
            dice_score = dice_coefficient_with_matching(targets[n]["masks"],
                                                        targets[n]["masks"],
                                                        targets[n]["boxes"],
                                                        targets[n]["boxes"],
                                                        0.5,
                                                        0.5)
            print(f"Dice: {dice_score.item()}")

    def test_component_mask_to_full(self):

        N = 8

        fig, ax = plt.subplots(nrows=N, ncols=3, figsize=(9, 3*N))

        for n in range(N):

            img, mask, _, _ = self.ds[np.random.randint(0, len(self.ds))]

            binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                                 num_classes=self.ds.num_classes,
                                                 ignore_index=self.ds.ignore_index,
                                                 keep_ignore_index=True)

            print(f"{binary_mask.shape=}")

            rgb_mask = convert_mask_to_RGB(mask=mask.unsqueeze(0),
                                           palette=self.ds.get_cmap(),
                                           ignore_index=self.ds.ignore_index)
            bb_list, label_list, mask_list = get_bb_from_mask(binary_mask)

            masks = mask_list[0]
            labels = label_list[0]

            padded_binary_mask = insert_component_masks(masks,
                                                        labels,
                                                        self.ds.num_classes,
                                                        self.ds.ignore_index)
            print(f"{labels=}")
            print(f"{padded_binary_mask.shape=}")
            padded_int_mask = convert_to_integer_mask(padded_binary_mask.unsqueeze(0),
                                                      self.ds.num_classes,
                                                      self.ds.ignore_index)
            print(f"{torch.unique(padded_int_mask)=}")
            print(f"{padded_int_mask.shape=}")
            padded_rgb_mask = convert_mask_to_RGB(padded_int_mask,
                                                  palette=self.ds.get_cmap(),
                                                  ignore_index=self.ds.ignore_index)

            print(f"{padded_rgb_mask.shape=}")
            print()

            ax[n, 0].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
            ax[n, 1].imshow(rgb_mask[0].permute(1, 2, 0).cpu().numpy())
            ax[n, 2].imshow(padded_rgb_mask[0].permute(1, 2, 0).cpu().numpy())

        plt.show()




