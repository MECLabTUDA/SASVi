import unittest

import torch
from torch.utils.data import DataLoader
import albumentations as A

from src.model import get_model_instance_segmentation
from src.data import prepare_data

from sds_playground.datasets import CaDISv2_Dataset


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.train_ds = CaDISv2_Dataset(
            root='/local/scratch/CaDISv2/',
            spatial_transform=None,
            img_normalization=A.Normalize(0.0, 1.0),
            exp=2,
            mode='train',
            filter_mislabeled=True,
            sample_sem_label=False
        )
        self.ignore_ids = [self.train_ds.ignore_index, 1, 5, 6]
        print(f"Ignoring ids: {self.ignore_ids}")

        num_train_classes = self.train_ds.num_classes - len(self.ignore_ids) + 1  # +1 for BG

        self.m = get_model_instance_segmentation(num_classes=num_train_classes,
                                                 hidden_ft=128,
                                                 trainable_backbone_layers=0).to('cuda')

    def test_overfit_1sample(self):
        optimizer = torch.optim.AdamW(self.m.parameters(),
                                      lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

        dl = DataLoader(self.train_ds, batch_size=1, num_workers=0, shuffle=True)

        images, masks, _, _ = next(iter(dl))
        images_list, targets = prepare_data(images, masks, self.train_ds, self.ignore_ids,
                                            device='cuda', shift_by_1=True)

        mask_rcnn_loss_weights = {
            'loss_classifier': 1.0,  # Weight for classification loss
            'loss_box_reg': 1.0,  # Weight for bounding box regression loss
            'loss_mask': 1.0,  # Weight for mask loss
            'loss_objectness': 1.0,  # Weight for objectness loss
            'loss_rpn_box_reg': 1.0  # Weight for RPN box regression loss
        }

        for step in range(1000):

            loss_dict = self.m(images_list, targets)

            # loss_sum = sum(loss for loss in loss_dict.values())
            loss_sum = sum(mask_rcnn_loss_weights[k] * loss for k, loss in loss_dict.items())

            print(f"Step {step} Loss {loss_sum}")

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()


        self.assertTrue(loss_sum < 0.1)
