import gc
import time

import cv2
import albumentations as A
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from sds_playground.datasets import Cataract1kSegmentationDataset

from src.model import get_model_instance_segmentation
from src.loss import *
from src.data import *
from src.utils import *


def train(epochs: int = 100,
          steps: int = -1,
          val_freq: int = 1,
          batch_size: int = 16,
          num_workers: int = 4,
          weighted_sampling: bool = False,
          weighted_loss: bool = False,
          initial_lr: float = 0.001,
          betas: tuple[float, float] = (0.9, 0.999),
          weight_decay: float = 0.01,
          scheduler_step_size: int = 100,
          scheduler_gamma: float = 0.5,
          img_size: tuple = (800, 800),
          img_norm: tuple = (0.0, 1.0),
          ignore_ids: list = [],
          shift_ids_by_1: bool = True,
          components: bool = True,
          min_comp_fraction: float = 0.0,
          hidden_ft: int = 256,
          backbone: str = 'ResNet50',
          trainable_backbone_layers=3,
          data_dir: Path = Path('local/scratch/Cataract-1k/'),
          log_dir: Path = Path('results/mask_rcnn_cataract1k/'),
          device: str = 'cuda'):

    exp_dir = log_dir / time.strftime('%Y-%m-%d_%H-%M')
    exp_dir.mkdir(parents=True, exist_ok=False)

    log = create_logger(exp_dir / 'logfile.log')
    log.info(f"{torch.cuda.is_available()=}")

    train_ds = Cataract1kSegmentationDataset(
        root=data_dir,
        spatial_transform=A.Compose([
            A.RandomResizedCrop(*img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-45, 45), p=0.5, border_mode=cv2.BORDER_CONSTANT,
                     value=(0., 0., 0.), mask_value=0),
            A.RandomBrightnessContrast(p=0.5),
        ]),
        img_normalization=A.Normalize(*img_norm),
        mode='train'
    )
    ignore_ids = [train_ds.ignore_index] + ignore_ids if train_ds.ignore_index is not None else ignore_ids
    log.info(f"Ignoring ids: {ignore_ids}")

    # 50% training data subset
    _train_ds, _ = random_split(train_ds, [len(train_ds) // 10, len(train_ds) - (len(train_ds) // 10)])

    num_train_classes = train_ds.num_classes - len(ignore_ids) + 1  # +1 for BG
    print(f"{num_train_classes=}")

    if weighted_sampling:
        sampler = WeightedRandomSampler(weights=compute_sample_weights(_train_ds, train_ds.num_classes),
                                        num_samples=len(_train_ds))
        train_dl = DataLoader(_train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_dl = DataLoader(_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = Cataract1kSegmentationDataset(
        root=data_dir,
        spatial_transform=A.Compose([
            A.Resize(*img_size)
        ]),
        img_normalization=A.Normalize(*img_norm),
        mode='val'
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Set Mask R-CNN loss to (custom) weighted loss
    if weighted_loss:
        loss_weights = compute_class_weights(_train_ds, train_ds.num_classes)
    else:
        loss_weights = False

    m = get_model_instance_segmentation(num_classes=num_train_classes,
                                        hidden_ft=hidden_ft,
                                        img_size=img_size,
                                        backbone=backbone,
                                        trainable_backbone_layers=trainable_backbone_layers,
                                        class_weights=loss_weights).to(device)

    mask_rcnn_loss_weights = {
        'loss_classifier': 1.0,  # Weight for classification loss
        'loss_box_reg': 1.0,  # Weight for bounding box regression loss
        'loss_mask': 1.0,  # Weight for mask loss
        'loss_objectness': 1.0,  # Weight for objectness loss
        'loss_rpn_box_reg': 1.0  # Weight for RPN box regression loss
    }

    optimizer = torch.optim.AdamW(m.parameters(), lr=initial_lr, betas=betas, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, power=0.2, total_iters=epochs)

    best_val_loss = 1e6
    best_val_iou = 0.0
    best_val_f1 = 0.0
    best_val_dice = 0.0

    #
    #   Training Loop
    #

    train_loss_per_ep = []
    val_loss_per_ep = []
    class_f1_per_ep = []
    bb_iou_per_ep = []
    mask_dice_per_ep = []
    for epoch in range(epochs):

        m.train()

        train_loss_dict = {"loss_classifier": 0, "loss_box_reg": 0, "loss_mask": 0, "loss_objectness": 0,
                           "loss_rpn_box_reg": 0}
        train_loss_sum = 0
        for step, (images, masks, _, _) in enumerate(tqdm(train_dl, desc=f'Training epoch {epoch}')):

            if step == steps:
                break

            images_list, targets = prepare_data(images, masks, train_ds, ignore_ids,
                                                device=device, shift_by_1=shift_ids_by_1,
                                                components=components, min_comp_fraction=min_comp_fraction)

            loss_dict = m(images_list, targets)
            for k, v in loss_dict.items():
                train_loss_dict[k] += v.item()

            # loss_sum = sum(loss for loss in loss_dict.values())
            loss_sum = sum(mask_rcnn_loss_weights[k] * loss for k, loss in loss_dict.items())
            train_loss_sum += loss_sum.item()

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            if (steps == -1 and step < len(train_dl) - 2) or (steps > 0 and step < steps - 1):
                del images_list, targets

        with torch.no_grad():
            m.eval()
            outputs = m(images_list)
            visualise_results(images_list,
                              targets,
                              outputs,
                              ds=train_ds,
                              ignored_indices=ignore_ids,
                              shift_by_1=shift_ids_by_1,
                              exp_dir=exp_dir,
                              device=device,
                              img_norm=img_norm,
                              name='train_examples.png')

        # Calculate average loss over all validation batches
        train_loss_sum /= (step + 1)

        # Calculate average for each individual loss component
        for k in train_loss_dict:
            train_loss_dict[k] /= (step + 1)

        # Update the learning rate
        lr_scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

        if (epoch + 1) % val_freq == 0:
            #
            #   Validation loop
            #

            val_loss_dict = {"loss_classifier": 0, "loss_box_reg": 0, "loss_mask": 0, "loss_objectness": 0,
                             "loss_rpn_box_reg": 0}
            val_loss_sum = 0
            with torch.no_grad():
                all_precisions = []
                all_recalls = []
                all_f1_scores = []
                all_iou_scores = []
                all_dice_scores = []
                for step, (images, masks, _, _) in enumerate(tqdm(val_dl, desc=f'Validation epoch {epoch}')):

                    images_list, targets = prepare_data(images, masks, val_ds, ignore_ids,
                                                        device=device, shift_by_1=shift_ids_by_1,
                                                        components=components, min_comp_fraction=min_comp_fraction)

                    # Forward pass in training mode to obtain validation losses
                    m.train()
                    loss_dict = m(images_list, targets)
                    for k, v in loss_dict.items():
                        val_loss_dict[k] += v.item()

                    # loss_sum = sum(loss for loss in loss_dict.values())
                    loss_sum = sum(mask_rcnn_loss_weights[k] * loss for k, loss in loss_dict.items())
                    val_loss_sum += loss_sum.item()

                    # Forward pass in eval mode to obtain inference results
                    m.eval()
                    outputs = m(images_list)

                    # Compute per-class metrics
                    metrics = compute_classification_metrics(targets, outputs)
                    iou_metrics = compute_class_iou(targets, outputs)
                    dice_metrics = compute_mask_dice_score(targets, outputs)

                    # Compute macro averages
                    macro_metrics = compute_macro_averages(metrics)
                    all_precisions.append(macro_metrics['macro_precision'])
                    all_recalls.append(macro_metrics['macro_recall'])
                    all_f1_scores.append(macro_metrics['macro_f1_score'])

                    macro_iou = compute_macro_iou(iou_metrics)
                    all_iou_scores.append(macro_iou['macro_average_iou'])

                    macro_dice = compute_macro_dice(dice_metrics)
                    all_dice_scores.append(macro_dice['macro_average_dice'])

                # Compute mean metrics over all validation data
                mean_iou = torch.tensor(all_iou_scores).mean().item() if len(all_iou_scores) > 0 else 0
                mean_precision = torch.tensor(all_precisions).mean().item() if len(all_precisions) > 0 else 0
                mean_recall = torch.tensor(all_recalls).mean().item() if len(all_recalls) > 0 else 0
                mean_f1 = torch.tensor(all_f1_scores).mean().item() if len(all_f1_scores) > 0 else 0
                mean_dice = torch.tensor(all_dice_scores).mean().item() if len(all_dice_scores) > 0 else 0

                # Calculate average loss over all validation batches
                val_loss_sum /= (step + 1)

                # Optionally, calculate the average for each individual loss component
                for k in val_loss_dict:
                    val_loss_dict[k] /= (step + 1)

                time.sleep(0.1)
                train_loss_per_ep.append(train_loss_sum)
                val_loss_per_ep.append(val_loss_sum)
                class_f1_per_ep.append(mean_f1)
                bb_iou_per_ep.append(mean_iou)
                mask_dice_per_ep.append(mean_dice)
                visualise_loss_and_metrics(train_loss_per_ep, val_loss_per_ep,
                                           class_f1_per_ep, bb_iou_per_ep, mask_dice_per_ep,
                                           epoch, val_freq, exp_dir)
                log.info(f"Epoch: {epoch}\t"
                         f"Train loss: {np.round(train_loss_sum, 3)}\t"
                         f"Val loss: {np.round(val_loss_sum, 3)}\t"
                         f"Val IoU: {np.round(mean_iou, 3)}\t"
                         f"Val F1: {np.round(mean_f1, 3)}\t"
                         f"Val Prec: {np.round(mean_precision, 3)}\t"
                         f"Val Rec: {np.round(mean_recall, 3)}\t"
                         f"Val Dice: {np.round(mean_dice, 3)}")

                #
                #   Checkpointing
                #

                if (val_loss_sum < best_val_loss) or (mean_iou > best_val_iou) or \
                        (mean_f1 > best_val_f1) or (mean_dice > best_val_dice):
                    visualise_results(images_list,
                                      targets,
                                      outputs,
                                      ds=train_ds,
                                      ignored_indices=ignore_ids,
                                      shift_by_1=shift_ids_by_1,
                                      exp_dir=exp_dir,
                                      device=device,
                                      img_norm=img_norm,
                                      name='val_examples.png')

                if val_loss_sum < best_val_loss:
                    best_val_loss = val_loss_sum
                    log.info("### New best validation loss :-) ###")
                    torch.save(m.state_dict(), os.path.join(exp_dir, 'best_val_loss.pth'))
                if mean_iou > best_val_iou:
                    best_val_iou = mean_iou
                    log.info("### New best validation BB IoU :-) ###")
                    torch.save(m.state_dict(), os.path.join(exp_dir, 'best_val_iou.pth'))
                if mean_f1 > best_val_f1:
                    best_val_f1 = mean_f1
                    log.info("### New best validation BB F1 :-) ###")
                    torch.save(m.state_dict(), os.path.join(exp_dir, 'best_val_f1.pth'))
                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    log.info("### New best validation Segm. Dice :-) ###")
                    torch.save(m.state_dict(), os.path.join(exp_dir, 'best_val_dice.pth'))

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train(epochs=1000,
          # steps=100,
          steps=-1,
          val_freq=5,
          batch_size=8,  # 8
          num_workers=4,
          weighted_sampling=True,  # True,
          weighted_loss=True,
          initial_lr=0.0001,
          betas=(0.5, 0.999),
          weight_decay=0.05,
          scheduler_step_size=200,
          scheduler_gamma=0.5,
          img_size=(299, 299),
          img_norm=(0.0, 1.0),
          ignore_ids=[],
          shift_ids_by_1=True,  # Must be true since we remove ignore_index=0
          components=False,
          min_comp_fraction=0.05,
          hidden_ft=32,
          backbone='ResNet18',
          trainable_backbone_layers=5,
          data_dir=Path('/home/yfrisch_locale/DATA/Cataract-1k/'),
          log_dir=Path('results/mask_rcnn___resnet18_32ft_5b'
                       '___cataracts1k_10perc___weighted_sampling___weighted_loss'
                       '___step_lr___no_comp___ignore_0/')
          )
