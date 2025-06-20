import argparse
import time
from pathlib import Path

import torch
import torchprofile
from torch.utils.data import DataLoader

from src.data import prepare_data
from src.utils import (load_maskrcnn_overseer, load_DETR_overseer, load_Mask2Former_overseer,
                       process_mask2former_outputs, process_detr_outputs, visualise_results)


def model_size(model):
    param_size = sum(p.numel() for p in model.parameters())
    #param_size *= 4 # 4 bytes per float32
    #return param_size / (1024 ** 2)  # Convert to MB
    return param_size


def estimate_flops(model, input_tensor):
    # Use torchprofile to estimate FLOPS
    flops = torchprofile.profile_macs(model, input_tensor)
    return flops


def eval(model: str, chckpt: Path, data: Path, device: str = 'cuda', visualise: bool = False):

    match model.upper():

        case 'MASKRCNN':
            m, ds, hidden_ft, backbone, ignore_indices, shift_by_1, keep_ignore = load_maskrcnn_overseer(chckpt, data, device)
        case 'DETR':
            m, processor, ds, num_train_classes, ignore_indices, keep_ignore = load_DETR_overseer(chckpt, data, device)
        case 'MASK2FORMER':
            m, ds, num_train_classes, ignore_indices, keep_ignore = load_Mask2Former_overseer(chckpt, data, device)
        case _:
            raise ValueError

    m.eval()

    dl = DataLoader(ds, batch_size=8, num_workers=1, drop_last=False, shuffle=False)

    images, masks, _, _ = next(iter(dl))

    images_list, targets = prepare_data(images, masks, ds, ignore_indices,
                                        device=device, shift_by_1=True, keep_ignore=keep_ignore,
                                        components=False, min_comp_fraction=0.0)

    # print(f"Model Size: {model_size(m):.2f} MB")
    print(f"\n\nModel Size: {model_size(m)} parameters")

    with torch.no_grad():

        match model.upper():
            case 'MASKRCNN':
                start_time = time.time()
                processed_outputs = m(images_list)
                end_time = time.time()
                #print(f"{type(images_list)=}")
                #print(f"{type(images_list[0])=}\\{images_list[0].shape=}")
                #exit()
                #flops = estimate_flops(m, (images_list[0].unsqueeze(0).to(device)))
            case 'DETR':
                inputs = processor(images=[image.cpu().numpy() for image in images_list],
                                   do_rescale=False,
                                   return_tensors="pt").to(device)
                start_time = time.time()
                outputs = m(**inputs)
                end_time = time.time()
                #flops = estimate_flops(m, **inputs)
                processed_outputs = process_detr_outputs(outputs,
                                                         (200, 200),
                                                         num_labels=num_train_classes,
                                                         threshold=0.0)
            case 'MASK2FORMER':
                start_time = time.time()
                outputs = m(
                    pixel_values=images.to(device),
                    mask_labels=[target["masks"].float() for target in targets],
                    class_labels=[target["labels"] for target in targets]
                )
                end_time = time.time()
                #flops = estimate_flops(m, images.to(device))
                processed_outputs = process_mask2former_outputs(outputs,
                                                                num_labels=num_train_classes,
                                                                image_size=(299, 299),
                                                                threshold=0.0)
            case _:
                raise ValueError

    inference_time = end_time - start_time
    print(f"Inference Time (single forward pass): {inference_time:.6f} seconds // {1/inference_time:.3f} FPS")

    #print(f"Estimated FLOPS: {flops / 1e9:.2f} GFLOPS")

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
            name=f'results/{ds.__class__.__name__}_efficiency.svg'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model type to evaluate.')
    parser.add_argument('--chckpt', type=str, help='Path to Mask R-CNN checkpoint.')
    parser.add_argument('--data', type=str, help='Path to data root.')
    parser.add_argument('--device', type=str, help='Device literal for inference.')
    parser.add_argument('--visualise', type=bool, default=False, help='Save predictions as plot.')
    args = parser.parse_args()
    eval(args.model, Path(args.chckpt), Path(args.data), device=args.device, visualise=args.visualise)
