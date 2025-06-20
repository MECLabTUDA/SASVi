import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
from torch.utils.data import DataLoader

from sds_playground.datasets import Cataract1kSegmentationDataset
from sds_playground.utils import *

from src.data import *
from src.utils import visualise_results


class Cataract1kDataTest(unittest.TestCase):

    def setUp(self):
        self.ds = Cataract1kSegmentationDataset(
            root=Path('/home/yfrisch_locale/DATA/Cataract-1k'),
            spatial_transform=A.Resize(800, 800),
            img_normalization=A.Normalize(0, 1),
            mode='full'
        )

    def testDatasetSize(self):
        print(len(self.ds))

    def test_convert_data(self):
        rand_ind = np.random.randint(0, len(self.ds))
        img, mask, _, _ = self.ds[rand_ind]

        print(f"{mask.shape=}")
        print(f"{torch.unique(mask)=}")
        for c in range(self.ds.num_classes):
            print(f"Class {c} Sum {torch.sum(mask == c)}")
        binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                             num_classes=self.ds.num_classes,
                                             ignore_index=self.ds.ignore_index,
                                             keep_ignore_index=False)
        print(f"{binary_mask.shape=}")
        for c in range(self.ds.num_classes - 1):
            print(f"Class {c} Sum {torch.sum(binary_mask[:, c, ...])}")

    def testConvert(self):
        dl = DataLoader(self.ds, batch_size=8, shuffle=True, num_workers=0)
        img, mask, _, _ = next(iter(dl))

        ignore_ids = [self.ds.ignore_index, 1, 2, 3]
        print(f"Ignoring ids: {ignore_ids}")

        images_list, targets = prepare_data(img, mask, self.ds, ignore_ids,
                                            device='cpu', shift_by_1=True, components=False)

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
            name='cataract1k_test.png'
        )

    def test_component_mask_to_full(self):
        N = 8

        fig, ax = plt.subplots(nrows=N, ncols=3, figsize=(9, 3*N))

        for n in range(N):

            rand_ind = np.random.randint(0, len(self.ds))
            img, mask, _, _ = self.ds[rand_ind]
            binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                                 num_classes=self.ds.num_classes,
                                                 ignore_index=self.ds.ignore_index,
                                                 keep_ignore_index=False)
            rgb_mask = convert_mask_to_RGB(mask=mask.unsqueeze(0),
                                           palette=self.ds.get_cmap(),
                                           ignore_index=self.ds.ignore_index)
            bb_list, label_list, mask_list = get_bb_from_mask(binary_mask)

            masks = mask_list[0]
            labels = label_list[0] + 1

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
