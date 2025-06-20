import gc
import time

import cv2
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

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
          backbone: str = 'ResNet50',
          num_queries: int = 100,
          data_dir: Path = Path('local/scratch/CholecSeg8k/'),
          log_dir: Path = Path('results/DETR_cholecseg/'),
          device: str = 'cuda'):

    exp_dir = log_dir / time.strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir.mkdir(parents=True, exist_ok=False)

    log = create_logger(exp_dir / 'logfile.log')
    log.info(f"{torch.cuda.is_available()=}")

    train_ds = CholecSeg8kDataset(
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

    # num_train_classes = train_ds.num_classes - len(ignore_ids) + 1  # +1 for BG
    num_train_classes = train_ds.num_classes - len(ignore_ids)  # TODO
    log.info(f"{num_train_classes=}")

    if weighted_sampling:
        sampler = WeightedRandomSampler(weights=compute_sample_weights(train_ds, train_ds.num_classes),
                                        num_samples=len(train_ds))
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = CholecSeg8kDataset(
        root=data_dir,
        spatial_transform=A.Compose([
            A.Resize(*img_size)
        ]),
        img_normalization=A.Normalize(*img_norm),
        mode='val'
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    match backbone.upper():
        case 'RESNET50':
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
            m = DetrForSegmentation.from_pretrained(
                "facebook/detr-resnet-50-panoptic",
                num_labels=num_train_classes,
                ignore_mismatched_sizes=True,
                num_queries=num_queries,
            ).to(device)
        case _:
            raise ValueError

    if weighted_loss:
        # TODO: Not working yet
        weight_dict = m.detr.weight_dict
        weight_dict['loss_ce'] = compute_class_weights(train_ds, train_ds.num_classes)

    optimizer = torch.optim.AdamW(m.parameters(), lr=initial_lr, betas=betas, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

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

        train_loss_sum = 0
        for step, (images, masks, _, _) in enumerate(tqdm(train_dl, desc=f'Training epoch {epoch}')):

            if step == steps:
                break

            images_list, targets = prepare_data(images, masks, train_ds, ignore_ids,
                                                device=device,
                                                shift_by_1=shift_ids_by_1,
                                                keep_ignore=True, # TODO: Argument only for integer -> binary conversion
                                                components=components, min_comp_fraction=min_comp_fraction)

            # Preprocess images using DETR processor
            inputs = processor(images=[image.cpu().numpy() for image in images_list],
                               do_rescale=False,
                               return_tensors="pt").to(device)

            detr_targets = create_detr_targets(targets, img_size)

            # Forward pass
            outputs = m(**inputs, labels=detr_targets)

            # loss_sum = sum(loss for loss in loss_dict.values())
            loss_sum = outputs.loss

            train_loss_sum += loss_sum.item()

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            if (steps == -1 and step < len(train_dl) - 2) or (steps > 0 and step < steps - 1):
                del images_list, targets

        with torch.no_grad():
            m.eval()
            outputs = m(**inputs)

            processed_outputs = process_detr_outputs(outputs,
                                                     img_size,
                                                     num_labels=num_train_classes,
                                                     # threshold=0.5)  # List of dict for each sample in the batch
                                                     threshold=0.0)  # List of dict for each sample in the batch

            visualise_results(images_list,
                              targets,
                              processed_outputs,
                              ds=train_ds,
                              ignored_indices=ignore_ids,
                              shift_by_1=shift_ids_by_1,
                              exp_dir=exp_dir,
                              device=device,
                              img_norm=img_norm,
                              name='train_examples.png')

        # Calculate average loss over all validation batches
        train_loss_sum /= (step + 1)

        # Update the learning rate
        lr_scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

        if (epoch + 1) % val_freq == 0:
            #
            #   Validation loop
            #

            val_loss_sum = 0
            m.eval()

            with torch.no_grad():


                all_precisions = []
                all_recalls = []
                all_f1_scores = []
                all_iou_scores = []
                all_dice_scores = []
                for step, (images, masks, _, _) in enumerate(tqdm(val_dl, desc=f'Validation epoch {epoch}')):

                    images_list, targets = prepare_data(images, masks, val_ds, ignore_ids,
                                                        device=device,
                                                        keep_ignore=True,
                                                        shift_by_1=shift_ids_by_1,
                                                        components=components, min_comp_fraction=min_comp_fraction)

                    # Preprocess images using DETR processor
                    inputs = processor(images=[image.cpu().numpy() for image in images_list],
                                       do_rescale=False,
                                       return_tensors="pt").to(device)

                    detr_targets = create_detr_targets(targets, img_size)

                    outputs = m(**inputs, labels=detr_targets)
                    # No thresholding to ensure fair comparison to Mask R-CNN
                    processed_outputs = process_detr_outputs(outputs,
                                                             img_size,
                                                             num_labels=num_train_classes,
                                                             # threshold=0.5)  # List of dict for each sample in the batch
                                                             threshold=0.0)  # List of dict for each sample in the batch

                    val_loss_sum += outputs.loss.item()

                    # Compute per-class metrics
                    metrics = compute_classification_metrics(targets, processed_outputs)
                    iou_metrics = compute_class_iou(targets, processed_outputs)
                    dice_metrics = compute_mask_dice_score(targets, processed_outputs)

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
                                      processed_outputs,
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
          steps=100,  # 100
          val_freq=5,  # 5
          batch_size=8,  # 4 // (8 for Mask R-CNN)
          num_workers=4,
          weighted_sampling=False,  # True
          weighted_loss=False,
          initial_lr=0.00001,
          betas=(0.9, 0.999),
          weight_decay=0.05,
          scheduler_step_size=200,
          scheduler_gamma=0.5,
          img_size=(200, 200),  # Matching DETR training data (299x299 for Mask R-CNN)
          img_norm=(0.0, 1.0),
          ignore_ids=[0],
          shift_ids_by_1=True,  # Must be true since we remove ignore_index=0
          components=False,
          min_comp_fraction=0.05,
          backbone='ResNet50',
          num_queries=20,
          data_dir=Path('/local/scratch/CholecSeg8k/'),
          log_dir=Path('results/DETR___20queries'
                       '___cholecseg___200pix___8bs'
                       '___step_lr___no_comp___ignore_0/')
          )
