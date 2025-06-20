import gc
import time

import cv2
import albumentations as A
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from sds_playground.datasets import Cataract1kSegmentationDataset

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
          backbone: str = 'swin-base-coco-panoptic',
          num_queries: int = 100,
          data_dir: Path = Path('local/scratch/Cataract-1k/'),
          log_dir: Path = Path('results/mask_rcnn_cataract1k/'),
          device: str = 'cuda'):

    exp_dir = log_dir / time.strftime('%Y-%m-%d_%H-%M-%S')
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

    num_train_classes = train_ds.num_classes - len(ignore_ids) + 1  # +1 for BG
    print(f"{num_train_classes=}")

    if weighted_sampling:
        sampler = WeightedRandomSampler(weights=compute_sample_weights(train_ds, train_ds.num_classes),
                                        num_samples=len(train_ds))
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = Cataract1kSegmentationDataset(
        root=data_dir,
        spatial_transform=A.Compose([
            A.Resize(*img_size)
        ]),
        img_normalization=A.Normalize(*img_norm),
        mode='val'
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # facebook/mask2former-swin-base-coco-panoptic
    processor = Mask2FormerImageProcessor.from_pretrained(
        f"facebook/mask2former-{backbone}",
        # ignore_index=ignore_ids,  # TODO: Explicitly providing ignore index
        reduce_labels=True
    )
    m = Mask2FormerForUniversalSegmentation.from_pretrained(
        f"facebook/mask2former-{backbone}",
        num_labels=num_train_classes,  # Adapt the number of classes
        ignore_mismatched_sizes=True,  # To adapt pretrained weights if num_classes changes,
        num_queries=num_queries  # 100 // TODO: Reduce?
    ).to(device)

    if weighted_loss:
        raise NotImplementedError
        loss_weights = compute_class_weights(train_ds, train_ds.num_classes)
        loss_weights = torch.cat([loss_weights, torch.Tensor([0.1])])  # Additional weight for 'no-obj' class
    else:
        loss_weights = None

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
                                                device=device, shift_by_1=shift_ids_by_1,
                                                components=components, min_comp_fraction=min_comp_fraction)

            # Preprocess images using DETR processor
            #inputs = processor(images=[image.cpu().numpy() for image in images_list],
            #                   do_rescale=True,  # False // TODO: Rescale to match backbone training image size?
            #                   return_tensors="pt").to(device)

            # Forward pass
            # outputs = m(**inputs)
            #print(f"{targets[0]['masks'].shape=}\t{torch.unique(targets[0]['masks'])=}")
            #print(f"{targets[0]['labels'].shape=}\t{torch.unique(targets[0]['labels'])=}")
            #print(f"{(targets[0]['masks'] * targets[0]['labels'].view(targets[0]['labels'].shape[0], 1, 1)).shape=}")
            #print(f"{torch.unique(targets[0]['masks'] * targets[0]['labels'].view(targets[0]['labels'].shape[0], 1, 1))=}")
            #print(f"{images.shape=}")
            #print(f"{type(inputs)=}")
            #print(f"{inputs[0].shape=}")
            outputs = m(
                pixel_values=images.to(device),
                mask_labels=[target["masks"].float() for target in targets],  # Binary objects' masks per sample in batch in (n_obj, H, W)
                # class_labels=[target["masks"] * target["labels"].view(target["labels"].shape[0], 1, 1) for target in targets],  # Integer objects' masks per sample in batch in (n_obj, H, W)
                class_labels=[target["labels"] for target in targets]
            )
            #print(f"{outputs.loss=}")
            #exit()
            """
            processed_outputs = process_mask2former_outputs(outputs,
                                                            num_labels=num_train_classes,
                                                            image_size=img_size,
                                                            threshold=-np.inf)  # Considering all predictions for loss computation

            total_loss = 0.0
            for n in range(len(images_list)):

                # Hungarian matching
                matches = hungarian_matching(processed_outputs[n]["logits"], processed_outputs[n]["logit_masks"],
                                             targets[n]["labels"], targets[n]["masks"])

                loss = mask2former_loss(processed_outputs[n]["logits"], processed_outputs[n]["logit_masks"],
                                        targets[n]["labels"], targets[n]["masks"],
                                        matches,
                                        classification_loss_fn=classification_loss_fn,
                                        mask_loss_fn=mask_loss_fn)

                total_loss += loss
            """
            total_loss = outputs.loss
            train_loss_sum += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            if (steps == -1 and step < len(train_dl) - 2) or (steps > 0 and step < steps - 1):
                del images_list, targets

        m.eval()
        with torch.no_grad():

            # outputs = m(**inputs)

            processed_outputs = process_mask2former_outputs(outputs,
                                                            num_labels=num_train_classes,
                                                            image_size=img_size,
                                                            # threshold=0.5)  # List of dict for each sample in the batch
                                                            threshold=0.0)  # List of dict for each sample in the batch

            # TODO: Non-maximum suppression (NMS)?

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
                                                        device=device, shift_by_1=shift_ids_by_1,
                                                        components=components, min_comp_fraction=min_comp_fraction)

                    # Preprocess images using DETR processor
                    #inputs = processor(images=[image.cpu().numpy() for image in images_list],
                    #                   do_rescale=False,
                    #                   return_tensors="pt").to(device)

                    # outputs = m(**inputs)

                    outputs = m(
                        pixel_values=images.to(device),
                        mask_labels=[target["masks"].float() for target in targets],
                        class_labels=[target["labels"] for target in targets]
                    )

                    """
                    # Considering all predictions for loss computation
                    processed_outputs = process_mask2former_outputs(outputs,
                                                                    num_labels=num_train_classes,
                                                                    image_size=img_size,
                                                                    threshold=-np.inf)  # List of dict for each sample in the batch

                    total_loss = 0.0
                    for n in range(len(images_list)):

                        # Hungarian matching
                        matches = hungarian_matching(processed_outputs[n]["logits"], processed_outputs[n]["logit_masks"],
                                                     targets[n]["labels"], targets[n]["masks"])

                        loss = mask2former_loss(processed_outputs[n]["logits"], processed_outputs[n]["logit_masks"],
                                                targets[n]["labels"], targets[n]["masks"],
                                                matches,
                                                classification_loss_fn=classification_loss_fn,
                                                mask_loss_fn=mask_loss_fn)

                        total_loss += loss

                    val_loss_sum += total_loss.item()
                    """
                    val_loss_sum += outputs.loss.item()

                    processed_outputs = process_mask2former_outputs(outputs,
                                                                    num_labels=num_train_classes,
                                                                    image_size=img_size,
                                                                    # threshold=0.5)
                                                                    threshold=0.0)

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

                    #processed_outputs = process_mask2former_outputs(outputs,
                    #                                                num_labels=num_train_classes,
                    #                                                image_size=img_size,
                    #                                                # threshold=0.5)  # List of dict for each sample in the batch
                    #                                                threshold=0.0)  # List of dict for each sample in the batch

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
          weighted_sampling=True,  # True
          weighted_loss=False,  # False
          initial_lr=0.0001,
          betas=(0.9, 0.999),
          weight_decay=0.05,
          scheduler_step_size=200,
          scheduler_gamma=0.5,
          img_size=(299, 299),  # Matching training data (299x299 for Mask R-CNN), (200x200 for DETR?)
          img_norm=(0.0, 1.0),
          ignore_ids=[],
          shift_ids_by_1=True,  # Must be true since we remove ignore_index=0
          components=False,
          min_comp_fraction=0.05,
          backbone='swin-base-coco-instance',
          num_queries=20,
          # data_dir=Path('/home/yfrisch_locale/DATA/Cataract-1k/'),
          data_dir=Path('/local/scratch/Cataract1kSegmentation/'),
          log_dir=Path('results/Mask2Former___swin_base_coco_instance___20queries'
                       '___cataracts1k___weighted_sampling___lr0_0001'
                       '___step_lr___no_comp___ignore_0/'),
          device='cuda'
          )
