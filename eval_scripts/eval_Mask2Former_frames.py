import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import prepare_data
from src.loss import *
from src.utils import load_Mask2Former_overseer, visualise_results, process_mask2former_outputs


def eval(chckpt: Path,
         data: Path,
         device: str = 'cuda',
         iou_threshold: float = 0.5,
         dice_threshold: float = 0.5,
         visualise: bool = False):

    img_size = (299, 299)

    m, ds, num_train_classes, ignore_indices, keep_ignore = load_Mask2Former_overseer(chckpt, data, device)

    dl = DataLoader(ds, batch_size=8, num_workers=4, drop_last=False, shuffle=False)

    all_precisions = []
    all_recalls = []
    all_class_f1_scores = []
    all_bb_iou_scores = []
    all_bb_dice_scores = []
    all_sem_dice_scores = []
    # unique_pred_values = Counter()
    unique_pred_values = set()
    # unique_gt_values = Counter()
    unique_gt_values = set()
    with torch.no_grad():
        for step, (images, masks, _, _) in enumerate(tqdm(dl, desc=f'Evaluating')):

            images_list, targets = prepare_data(images, masks, ds, ignore_indices,
                                                device=device, shift_by_1=True, keep_ignore=keep_ignore,
                                                components=False, min_comp_fraction=0.0)
                                                # components=True, min_comp_fraction=0.05)

            outputs = m(
                pixel_values=images.to(device),
                mask_labels=[target["masks"].float() for target in targets],
                class_labels=[target["labels"] for target in targets]
            )

            processed_outputs = process_mask2former_outputs(outputs,
                                                            num_labels=num_train_classes,
                                                            image_size=img_size,
                                                            threshold=0.0)

            if visualise:
                visualise_results(
                    images_list=images_list,
                    outputs=processed_outputs,
                    targets=targets,
                    ds=ds,
                    ignored_indices=ignore_indices,
                    shift_by_1=True,
                    img_norm=(0.0, 1.0),
                    device='cpu',
                    exp_dir=Path('./'),
                    remove_overlap=True,
                    target_size=ds.display_shape[1:],
                    name=f'results/{ds.__class__.__name__}_inference_step{step}.svg'
                )

            # Compute per-class metrics
            metrics = compute_classification_metrics(targets, processed_outputs)
            iou_metrics = compute_class_iou(targets, processed_outputs)
            dice_metrics = compute_mask_dice_score(targets, processed_outputs)

            # Compute macro averages
            macro_metrics = compute_macro_averages(metrics)
            all_precisions.append(macro_metrics['macro_precision'])
            all_recalls.append(macro_metrics['macro_recall'])
            all_class_f1_scores.append(macro_metrics['macro_f1_score'])

            macro_iou = compute_macro_iou(iou_metrics)
            all_bb_iou_scores.append(macro_iou['macro_average_iou'])

            macro_dice = compute_macro_dice(dice_metrics)
            all_bb_dice_scores.append(macro_dice['macro_average_dice'])

            # Iterate over each image in the batch
            for i, output in enumerate(processed_outputs):

                gt_boxes = targets[i]["boxes"]
                pred_boxes = output["boxes"]
                gt_masks = targets[i]["masks"]
                pred_masks = output["masks"].squeeze(1)

                # TODO
                unique_pred_values.update(output["labels"].flatten().tolist())
                unique_gt_values.update(targets[i]["labels"].flatten().tolist())

                if len(pred_masks) > 0 and len(gt_masks) > 0:
                    # Compute Dice coefficient for masks
                    dice_score = dice_coefficient_with_matching(pred_masks,
                                                                gt_masks,
                                                                pred_boxes,
                                                                gt_boxes,
                                                                iou_threshold,
                                                                dice_threshold)
                    all_sem_dice_scores.append(dice_score.item())

    print(f"{unique_pred_values=}")
    print(f"{unique_gt_values=}")
    print(f"{num_train_classes=}")
    print(f"{ds.num_classes=}")
    exit()

    # Compute mean metrics over all validation data
    mean_precision = torch.tensor(all_precisions).mean().item() if len(all_precisions) > 0 else 0
    mean_recall = torch.tensor(all_recalls).mean().item() if len(all_recalls) > 0 else 0
    mean_class_f1 = torch.tensor(all_class_f1_scores).mean().item() if len(all_class_f1_scores) > 0 else 0
    mean_bb_iou = torch.tensor(all_bb_iou_scores).mean().item() if len(all_bb_iou_scores) > 0 else 0
    mean_bb_dice = torch.tensor(all_bb_dice_scores).mean().item() if len(all_bb_dice_scores) > 0 else 0
    mean_sem_dice = torch.tensor(all_sem_dice_scores).mean().item() if len(all_sem_dice_scores) > 0 else 0

    print(f"Class F1: {np.round(mean_class_f1, 3)}\t"
          f"Prec.: {np.round(mean_precision, 3)}\t"
          f"Rec.: {np.round(mean_recall, 3)}\t"
          f"BB IoU: {np.round(mean_bb_iou, 3)}\t"
          f"BB Dice: {np.round(mean_bb_dice, 3)}\t"
          f"Sem. Dice: {np.round(mean_sem_dice, 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chckpt', type=str, help='Path to Mask R-CNN checkpoint.')
    parser.add_argument('--data', type=str, help='Path to data root.')
    parser.add_argument('--device', type=str, help='Device literal for inference.')
    parser.add_argument('--visualise', type=bool, default=False, help='Save predictions as plot.')
    args = parser.parse_args()

    eval(Path(args.chckpt), Path(args.data), device=args.device, visualise=args.visualise)
