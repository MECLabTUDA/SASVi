from pathlib import Path
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm

from sds_playground.datasets import CaDISv2_Dataset
from sds_playground.datasets.cadisv2.cadisv2_experiments import EXP2
from sds_playground.utils import convert_to_binary_mask, convert_to_integer_mask, convert_mask_to_RGB

from src.data import get_bb_from_mask, insert_component_masks, prepare_data
from src.utils import visualise_results


class CadisDataTest(unittest.TestCase):

    def setUp(self):
        self.size = (299, 299)

        self.ds = CaDISv2_Dataset(
            root=Path('/local/scratch/CaDISv2/'),
            exp=2,
            spatial_transform=A.Compose([
                A.RandomResizedCrop(*self.size, scale=(0.9, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]),
            img_normalization=A.Normalize(0, 1),
            filter_mislabeled=True,
            mode='train'
        )
        self.val_ds = CaDISv2_Dataset(
            root=Path('/local/scratch/CaDISv2/'),
            exp=2,
            spatial_transform=A.Resize(*self.size),
            img_normalization=A.Normalize(0, 1),
            filter_mislabeled=True,
            mode='val'
        )
        self.test_ds = CaDISv2_Dataset(
            root=Path('/local/scratch/CaDISv2/'),
            exp=2,
            spatial_transform=A.Resize(*self.size),
            img_normalization=A.Normalize(0, 1),
            filter_mislabeled=True,
            mode='test'
        )

    def test_data_size(self):
        print(len(self.ds) + len(self.val_ds) + len(self.test_ds))

    def test_data_load(self):
        train_dl = DataLoader(self.ds, batch_size=8, num_workers=4, shuffle=True)
        for _ in tqdm(train_dl, desc='Training data'):
            pass
        val_dl = DataLoader(self.val_ds, batch_size=8, num_workers=4, shuffle=False)
        for _ in tqdm(val_dl, desc='Validation data'):
            pass
        test_dl = DataLoader(self.test_ds, batch_size=8, num_workers=4, shuffle=False)
        for _ in tqdm(test_dl, desc='Test data'):
            pass
        self.assertTrue(True)

    def test_data_load_conversion(self):
        train_dl = DataLoader(self.ds, batch_size=8, num_workers=4, shuffle=True)
        for img, mask, _, _ in tqdm(train_dl, desc='Training data'):
            _, _ = prepare_data(img, mask, self.ds, ignore_indices=[self.ds.ignore_index, 1, 5, 6], shift_by_1=True)
        val_dl = DataLoader(self.val_ds, batch_size=8, num_workers=4, shuffle=False)
        for img, mask, _, _ in tqdm(val_dl, desc='Validation data'):
            _, _ = prepare_data(img, mask, self.val_ds, ignore_indices=[self.ds.ignore_index, 1, 5, 6], shift_by_1=True)
        test_dl = DataLoader(self.test_ds, batch_size=8, num_workers=4, shuffle=False)
        for img, mask, _, _ in tqdm(test_dl, desc='Test data'):
            _, _ = prepare_data(img, mask, self.test_ds, ignore_indices=[self.ds.ignore_index, 1, 5, 6],
                                shift_by_1=True)

        self.assertTrue(True)

    def test_data_types(self):
        rand_ind = np.random.randint(0, len(self.ds))
        img, mask, _, _ = self.ds[rand_ind]
        self.assertEqual(type(img), torch.Tensor)
        self.assertEqual(type(mask), torch.Tensor)

    def test_data_shapes(self):
        rand_ind = np.random.randint(0, len(self.ds))
        img, mask, _, _ = self.ds[rand_ind]
        self.assertEqual(img.shape[-2:], mask.shape[-2:])
        self.assertEqual(list(img.shape), [3, *self.size])
        self.assertEqual(list(mask.shape), [*self.size])

    def test_to_binary_mask(self):
        rand_ind = np.random.randint(0, len(self.ds))
        _, mask, _, _ = self.ds[rand_ind]
        binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                             num_classes=self.ds.num_classes,
                                             ignore_index=self.ds.ignore_index,
                                             keep_ignore_index=False)
        self.assertEqual(list(binary_mask.shape), [1, self.ds.num_classes - 1, *self.size])

    def test_to_bb_shapes(self):
        rand_ind = np.random.randint(0, len(self.ds))
        img, mask, _, _ = self.ds[rand_ind]
        binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                             num_classes=self.ds.num_classes,
                                             ignore_index=self.ds.ignore_index,
                                             keep_ignore_index=False)
        bb_list, label_list, mask_list = get_bb_from_mask(binary_mask)
        bounding_boxes = bb_list[0]
        labels = label_list[0]
        masks = mask_list[0]

        self.assertTrue(bounding_boxes.shape[0] == labels.shape[0] == masks.shape[0])

    def test_to_bb_vis(self):
        rand_ind = np.random.randint(0, len(self.ds))
        img, mask, _, _ = self.ds[rand_ind]
        binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                             num_classes=self.ds.num_classes,
                                             ignore_index=self.ds.ignore_index,
                                             keep_ignore_index=False)
        bb_list, label_list, mask_list = get_bb_from_mask(binary_mask)
        bounding_boxes = bb_list[0]
        labels = label_list[0]
        masks = mask_list[0]

        rgb_mask = convert_mask_to_RGB(mask=mask.unsqueeze(0),
                                       palette=self.ds.get_cmap(),
                                       ignore_index=self.ds.ignore_index)

        label_names = EXP2["CLASS"]
        fig, ax = plt.subplots(nrows=3, ncols=labels.shape[0], figsize=(masks.shape[0] * 3, 9))
        ax[0, 0].imshow(img.permute(1, 2, 0).cpu().numpy())
        ax[1, 0].imshow(rgb_mask.squeeze(0).permute(1, 2, 0).cpu().numpy())
        for c in range(masks.shape[0]):
            ax[0, c].axis('off')
            ax[1, c].axis('off')
            ax[2, c].axis('off')
            ax[2, c].imshow(masks[c].cpu().numpy())
            ax[2, c].set_title(label_names[labels[c].item()])
        plt.tight_layout()
        plt.show()

    def testConvert(self):
        dl = DataLoader(self.ds, batch_size=8, shuffle=True, num_workers=0)
        img, mask, _, _ = next(iter(dl))

        ignore_ids = [self.ds.ignore_index]
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
            name='cadisv2_test.png'
        )

    def test_component_mask_to_full(self):

        N = 8

        fig, ax = plt.subplots(nrows=N, ncols=3, figsize=(9, 3 * N))

        for n in range(N):

            img, mask, _, _ = self.ds[np.random.randint(0, len(self.ds))]

            binary_mask = convert_to_binary_mask(mask.unsqueeze(0),
                                                 num_classes=self.ds.num_classes,
                                                 ignore_index=self.ds.ignore_index,
                                                 keep_ignore_index=False)

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
            print(f"{padded_binary_mask.shape=}")

            padded_int_mask = convert_to_integer_mask(padded_binary_mask.unsqueeze(0),
                                                      self.ds.num_classes,
                                                      self.ds.ignore_index)
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
