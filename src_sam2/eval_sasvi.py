#TODO: Even over here there can be reversing (If new object detected but still blurry!) - If the tool exist continuosly in few frames, you can prompt it in the middle, then reverse it (might not be suitable for cholec) 
#TODO: Prediction on reverse direction when mmaskrcnn prediction confidence is high
#TODO: Only tracking new object detected (Instead of new bounding box detected)
#TODO: Overall cleanup on V3 implementation
#TODO: Explore using result from nnunet when maskrcnn prediction is close (On the area of duplicate find which label has majority)
#TODO: Fine-tune SAM2 on the dataset
#TODO: Also remove label of tools leaving the scene 

import os
import sys
import numpy as np
import torch
import argparse
import itertools
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import find_objects
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import build_sam2_video_predictor
from src.sam2 import kmeans_sampling
from eval_sam2 import load_overseer


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item) 
        else:
            yield item 


def load_ann_png(path, 
                shift_by_1,
                reshape_size=None, 
                convert_to_label=False, 
                palette=None
                ):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    if reshape_size is not None:
        mask = mask.resize(reshape_size, Image.NEAREST)
    mask = np.array(mask).astype(np.uint8)

    if convert_to_label:
        if palette is None:
            raise ValueError("palette is required to convert the mask to label")
        else:
            mask_reshaped = mask.reshape(-1, 3)
            #TODO: Need to handle ignore and background better
            if shift_by_1:
                output = np.full(mask_reshaped.shape[0], 255, dtype=np.uint8)
            else:
                output = np.full(mask_reshaped.shape[0], 0, dtype=np.uint8)
            for i, color in enumerate(palette):
                matches = np.all(mask_reshaped == color, axis=1)
                output[matches] = i
            mask = output.reshape(mask.shape[:2])
    return mask


def load_ann_npz(path, 
                 reshape_size=None,
                 ignore_indices=None,
                 ):
    """Load a NPZ file as a mask."""
    npz_mask = np.load(path, allow_pickle=True)
    npz_mask = npz_mask["arr"].astype(bool)
    if reshape_size is not None:
        mask = np.array([resize(mask, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for mask in npz_mask])
    else:
        mask = npz_mask.astype(bool)
    #TODO: Need to handle ignore and background better
    if len(ignore_indices) > 0:
        length = len(ignore_indices)
        mask = mask[:-length]
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
    elif len(ignore_indices) == 0:
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
        object_ids = object_ids[1:]

    per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}
    return per_obj_mask


def save_ann_png(path, mask, palette, reshape_size=None):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    if reshape_size is not None:
        output_mask = output_mask.resize(reshape_size, Image.NEAREST)
    output_mask.putpalette(np.uint8(palette))
    output_mask.save(path)


def get_per_obj_mask(mask_path, frame_name, dataset_type, use_binary_maskrcnn, width, height, ignore_indices, shift_by_1, palette):
    """Split a mask into per-object masks."""
    # For CholecSeg8K, the gt mask and frame are saved according to frame rate. But maskrcnn paths are not according to frame rate.
    if not use_binary_maskrcnn:
        mask = load_ann_png(path=os.path.join(mask_path, f"{frame_name}.png"),
                            shift_by_1=shift_by_1,
                            reshape_size=(width, height),
                            palette=palette)
        object_ids = np.unique(mask)
        object_ids = np.array([item for item in object_ids if item not in ignore_indices])
        object_ids = object_ids[object_ids >= 0].tolist()
        per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    else:
        if width is not None and height is not None: reshape_size=(height, width)
        else: reshape_size = None

        if dataset_type == "CHOLECSEG8K":
            per_obj_mask = load_ann_npz(path=os.path.join(mask_path, f"{str(int(int(frame_name) / 2)).zfill(10)}_binary_mask.npz"), 
                                        reshape_size=reshape_size, 
                                        ignore_indices=ignore_indices)
        else:
            per_obj_mask = load_ann_npz(path=os.path.join(mask_path, f"{frame_name}_binary_mask.npz"), 
                                        reshape_size=reshape_size, 
                                        ignore_indices=ignore_indices)
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width, shift_by_1):
    """Combine per-object masks into a single mask."""
    #TODO: Need to handle ignore and background better 
    if shift_by_1:
        mask = np.full((height, width), 255, dtype=np.uint8)
    else:  
        mask = np.full((height, width), 0, dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def get_bbox_from_mask(mask):
    """Get the bounding box from gt mask."""
    unique_labels = np.unique(mask)
    for label in unique_labels:
        label = int(label)
        binary_mask = (mask == label)
        # Get the bounding box
        slices = find_objects(binary_mask)[0]
        bounding_box = (slices[0].start, slices[0].stop, slices[1].start, slices[1].stop)
        bbox = {label: bounding_box}
    return bbox


def get_points_from_mask(mask, label_ids=None, num_points=20):
    """Get the points from gt mask."""
    label_ids = label_ids if label_ids is not None else set(mask.keys())
    for label in label_ids:
        binary_mask = mask[label]
        # Get the prompt
        try:
            points = kmeans_sampling(torch.tensor(np.argwhere(binary_mask)), num_points)
            points = points.cpu().numpy().tolist()
            points = [[y, x] for x, y in points]
        except:
            points = []
        prompt_points = {label: points}
    return prompt_points


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_palette,
    save_binary_mask,
    num_classes,
    shift_by_1,
    save_height=299,
    save_width=299,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_mask = put_per_obj_mask(per_obj_output_mask, height, width, shift_by_1)
    output_mask_path = os.path.join(
        output_mask_dir, video_name, f"{frame_name}_rgb_mask.png"
    )
    save_ann_png(output_mask_path, output_mask, output_palette, reshape_size=(save_width, save_height))

    if save_binary_mask:
        for i in range(num_classes):
            if i not in per_obj_output_mask:
                per_obj_output_mask[i] = np.full((1, height, width), False)
        output_mask = dict(sorted(per_obj_output_mask.items()))
        output_mask = np.array(list(output_mask.values()))
        output_mask = np.squeeze(output_mask, axis=None)
        #TODO: Need to handle ignore and background better
        if shift_by_1:
            output_mask[-1] |= ~output_mask[:-1].any(axis=0)
        else:
            output_mask[0] |= ~output_mask[1:].any(axis=0)

        reshape_size = (save_height, save_width)
        output_mask = np.array([resize(mask, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for mask in output_mask])
        output_mask_path = os.path.join(output_mask_dir, video_name, f"{frame_name}_binary_mask.npz")
        np.savez_compressed(file=output_mask_path, arr=output_mask)


def get_unique_label(per_obj_mask_n):
    unique_lable_n = []
    for per_obj_mask in per_obj_mask_n:
        unique_lable_n.append(sorted(set(per_obj_mask.keys())))
    return unique_lable_n


def choose_duplicate_label(per_obj_mask_n, duplicate_label):
    selections = {}
    for per_obj_mask in per_obj_mask_n:
        unique_label = sorted(set(per_obj_mask.keys()))

        # if only one label from duplicate exist in next frame
        matching = list(set(duplicate_label) & set(unique_label))
        if len(matching) == 0:
            continue
        if len(matching) == 1:
            mask_size = np.sum((per_obj_mask[matching[0]] == 1))
            if matching[0] in selections:
                selections[matching[0]] += mask_size
            else: selections[matching[0]] = mask_size
        # if both or more labels from duplicate exist in next frame 
        else:
            temp_selections = []
            for pair in list(itertools.combinations(matching, 2)):
                mask1 = per_obj_mask[pair[0]]
                mask2 = per_obj_mask[pair[1]]
                true_positions_1 = (mask1 == 1)                                                                                                                
                true_positions_2 = (mask2 == 1)
                matching_true_positions = np.logical_and(true_positions_1, true_positions_2)
                similarity_1 = np.sum(matching_true_positions) / np.sum(true_positions_1)
                similarity_2 = np.sum(matching_true_positions) / np.sum(true_positions_2)

                if similarity_1 >= 0.70 and similarity_2 >= 0.70:
                    temp_selections.append(pair[0])
                    temp_selections.append(pair[1])
            
            for item in list(dict.fromkeys(temp_selections)):
                mask_size = np.sum((per_obj_mask[item] == 1))
                if item in selections:
                    selections[item] += mask_size
                else: selections[item] = mask_size

    return max(selections, key=selections.get)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def sasvi_inference(
    predictor,
    base_video_dir,
    gt_mask_dir,
    maskrcnn_mask_dir,
    output_mask_dir,
    video_name,
    maskrcnn_model,

    num_classes,
    ignore_indices,
    shift_by_1,
    palette,
    dataset_type,

    prediction_type="V3",
    score_thresh=0.0,
    save_binary_mask=False,
):
    """Run inference on a single video with the given predictor."""
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    if prediction_type == "V2" or prediction_type == "V3":
        gt_mask_frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(os.path.join(gt_mask_dir, video_name))
            if os.path.splitext(p)[-1] in [".png", ".PNG"]
        ]
    else: 
        gt_mask_frame_names = []

    if dataset_type == "CADIS":
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][5:]))
        if prediction_type == "V2" or prediction_type == "V3": gt_mask_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][5:]))
    else:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if prediction_type == "V2" or prediction_type == "V3": gt_mask_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    #TODO: Need to handle this better
    # bcs maskrcnn prediction didnt have it for last frame 
    frame_names = frame_names[:-1]

    # load the video frames and initialize the inference state of SAM2 on this video
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    gt_mask_found_flag = False
    stop_nested_loop = False
    break_endless_loop = False

    # run propagation throughout the video and collect the results in a dict
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)

    idx = 0
    while idx < len(frame_names):
        # clear the prompts from previous runs
        print("New inference for segment: " + frame_names[idx])
        predictor.reset_state(inference_state=inference_state)
        buffer_length = 25

        # this loads the masks. will add those input masks to SAM 2 inference state before propagation
        #TODO: Hotfixing, clean up the get_per_obj_mask reading duplicate
        if idx == 0:
            per_obj_input_mask = get_per_obj_mask(
                                mask_path=os.path.join(maskrcnn_mask_dir, video_name), 
                                frame_name=frame_names[idx], 
                                dataset_type=dataset_type, 
                                use_binary_maskrcnn=True,
                                width=width, 
                                height=height,
                                ignore_indices=ignore_indices, 
                                shift_by_1=shift_by_1, 
                                palette=palette)                     
            
        if idx > 0 and gt_mask_found_flag:
            if frame_names[idx] in gt_mask_frame_names:
                stop_nested_loop = True
                per_obj_input_mask = get_per_obj_mask(
                                        mask_path=os.path.join(gt_mask_dir, video_name), 
                                        frame_name=frame_names[idx], 
                                        dataset_type=dataset_type, 
                                        use_binary_maskrcnn=False, 
                                        width=width, 
                                        height=height,
                                        ignore_indices=ignore_indices, 
                                        shift_by_1=shift_by_1, 
                                        palette=palette)
            old_label = sorted(set(per_obj_input_mask.keys()))
            # old_label = sorted(set(per_obj_mask.keys()))

        elif idx > 0 and prediction_type == "V3":
            per_obj_input_mask = get_per_obj_mask(
                                    mask_path=os.path.join(maskrcnn_mask_dir, video_name), 
                                    frame_name=frame_names[idx], 
                                    dataset_type=dataset_type, 
                                    use_binary_maskrcnn=True,
                                    width=width, 
                                    height=height,
                                    ignore_indices=ignore_indices, 
                                    shift_by_1=shift_by_1, 
                                    palette=palette)
            
            # to make it more stable, if something ignored in maskrcnn frame, but detected in previous sam2 frame, add it to the prompt mask
            per_obj_previous_mask = get_per_obj_mask(
                                        mask_path=os.path.join(output_mask_dir, video_name), 
                                        frame_name=frame_names[idx],
                                        # Manually defining CADIS (including for CholecSeg), to avoid this index renaming for cholec maskrcnn prediction
                                        dataset_type="CADIS",
                                        use_binary_maskrcnn=True, 
                                        width=width, 
                                        height=height,
                                        ignore_indices=ignore_indices, 
                                        shift_by_1=shift_by_1, 
                                        palette=palette)

            if dataset_type == "CADIS":
                #TODO: Maybe there is more effeicient way to do this
                ignore_class_input_mask = ~np.array(list(per_obj_input_mask.values())).any(axis=0)
                ignore_class_previous_mask = ~np.array(list(per_obj_previous_mask.values())).any(axis=0)
                additional_points = np.argwhere(ignore_class_input_mask & ~ignore_class_previous_mask)
                merged_previous_mask = put_per_obj_mask(per_obj_previous_mask, height, width, shift_by_1)
                for pos in additional_points:
                    val = merged_previous_mask[tuple(pos)]
                    if val in list(set(old_label) & set(current_label)):
                        per_obj_input_mask[val][tuple(pos)] = True

            for obj_id in prompt_label_list:
                # adding positive points for new objects
                selected_point = get_points_from_mask(per_obj_input_mask, label_ids=[obj_id])
                if selected_point[obj_id]:
                    points = np.array(selected_point[obj_id], dtype=np.float32)
                    labels = np.ones((len(selected_point[obj_id]),), dtype=np.int32) 
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )

                # since mask have inaccurate label at positions where new tool is occupied, modify those areas to be false        
                false_positions = np.argwhere(per_obj_input_mask[obj_id])
                for obj_input_mask in per_obj_input_mask:
                    # sometimes sam2 is already correctly guessing them, so we dont want to disable those areas
                        if obj_input_mask != obj_id:
                            for false_pos in false_positions:
                                per_obj_input_mask[obj_input_mask][tuple(false_pos)] = False

            # update the old label 
            yet_another_unique_label = old_label
            break_endless_loop = True
            negative_duplicate_list = [x for xs in negative_duplicate_list for x in xs]
            old_label = list((((set(old_label) & set(current_label)) | set(prompt_label_list)) - set(negative_duplicate_list)))

        # add the corrected mask to predictor
        for object_id, object_mask in per_obj_input_mask.items():
            #TODO: Hotfixed the if statement in 396 and 405 (Can be better)
            if idx == 0 or idx > 0 and gt_mask_found_flag:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx,
                    obj_id=object_id,
                    mask=object_mask,
                )

            elif idx > 0 and prediction_type == "V3" and not gt_mask_found_flag:
                if object_id in yet_another_unique_label and object_id not in negative_duplicate_list:
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=idx,
                        obj_id=object_id,
                        mask=object_mask,
                    )
        
        if len(inference_state['point_inputs_per_obj']) == 0 and len(inference_state['mask_inputs_per_obj']) == 0:
            print("Empty maskrcnn mask, using dummy background point prompt. Happens when camera move out of scene.")
            dummy_id = num_classes if shift_by_1 else 0
            dummy_mask = np.zeros((height, width), dtype=bool)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=idx,
                obj_id=dummy_id,
                mask=dummy_mask,
            )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state): 
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            # write the output masks as palette PNG files to output_mask_dir
            save_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                output_palette=palette,
                save_binary_mask=save_binary_mask,
                num_classes=num_classes,
                shift_by_1=shift_by_1,
            )
            if idx > 0 and frame_names[idx] in gt_mask_frame_names:
                gt_mask_found_flag = True
                if stop_nested_loop: 
                    stop_nested_loop = False
                    gt_mask_found_flag = False
                    idx += 1
                    continue
                else: 
                    print("GT mask found for: " + frame_names[idx])
                    break

            elif prediction_type == "V3":
                #TODO: Define the length as parameter
                future_n_frame = min(10, len(frame_names) - idx)
                per_obj_input_mask_n = []
                duplicate_list = []
                negative_duplicate_list = []
                partial_duplicate_list = []
                prompt_label_list = []

                #TODO: Reading the n mask everytime, can be optimized
                for n in range(future_n_frame):
                    per_obj_input_mask_n.append(get_per_obj_mask(
                        mask_path=os.path.join(maskrcnn_mask_dir, video_name), 
                        frame_name=frame_names[idx+n], 
                        dataset_type=dataset_type, 
                        use_binary_maskrcnn=True, 
                        width=None, 
                        height=None,
                        ignore_indices=ignore_indices, 
                        shift_by_1=shift_by_1,
                        palette=palette))
                    
                unique_lable_n = get_unique_label(per_obj_input_mask_n)
                if idx == 0:
                    old_label = unique_lable_n[0]
                else:
                    current_label = unique_lable_n[0]
                    
                    # to restart the inference after 50 frames to avoid false labels to continue longer
                    buffer_length -= 1                    
                    if buffer_length == 0:
                        break
                    
                    if old_label != current_label:
                        # only if current label have new objects compared to old label (when tool entering scene). But not when a tool exiting!
                        # also ignoring labels that appear in one frame but not in next 3 frames
                        #TODO: Can define this as parameter and be more dynamic
                        new_obj_label = list(set(current_label) - set(old_label))
                        unique_lable_n = unique_lable_n[1:4]
                        for unique_l in unique_lable_n:
                            new_obj_label = list(set(new_obj_label) & set(unique_l))
                        
                        if new_obj_label:
                            # find out exact duplicates of new_obj_label, and run analysis on them
                            for new_obj in new_obj_label:
                                mask1 = per_obj_input_mask_n[0][new_obj]
                                for obj in current_label:
                                    mask2 = per_obj_input_mask_n[0][obj]

                                    true_positions_1 = (mask1 == 1)
                                    true_positions_2 = (mask2 == 1)
                                    matching_true_positions = np.logical_and(true_positions_1, true_positions_2)
                                    similarity_1 = np.sum(matching_true_positions) / np.sum(true_positions_1)
                                    similarity_2 = np.sum(matching_true_positions) / np.sum(true_positions_2)

                                    # true duplicate
                                    if similarity_1 >= 0.70 and similarity_2 >= 0.70 and obj != new_obj:
                                        if not duplicate_list:
                                            duplicate_list.append([obj, new_obj])
                                        else:
                                            common_element_flag = False
                                            for sublist in duplicate_list:
                                                if list(set(sublist) & set([obj, new_obj])):
                                                    common_element_flag = True
                                                    new_element = list(set([obj, new_obj]) - set(sublist))
                                                    if new_element: sublist.append(new_element[0])
                                            if not common_element_flag:
                                                duplicate_list.append([obj, new_obj])
                                    
                                    # if one similarity is high but not other way, partial duplicate and remove the smaller label
                                    elif similarity_1 >= 0.70 and similarity_2 < 0.70:
                                        partial_duplicate_list.append(new_obj)
                            
                            # if new duplicate is being replaced, we need to remove it (Kinda doing this with false_positions, but need to be more explicit)                            
                            if duplicate_list:
                                negative_duplicate_list = duplicate_list
                                for sublist in duplicate_list:
                                    #TODO: If score too similar, prefer label with existing one
                                    prompt_label = choose_duplicate_label(per_obj_mask_n=per_obj_input_mask_n, duplicate_label=sublist)
                                    if prompt_label not in old_label:
                                        prompt_label_list.append(prompt_label)
                                    negative_duplicate_list = [[x for x in slist if x != prompt_label] for slist in negative_duplicate_list]
                                    
                            if new_obj_label:
                                single_label = list((set(new_obj_label) - set([x for xs in duplicate_list for x in xs])) - set(partial_duplicate_list))
                                if single_label:
                                    for ixn in single_label:
                                        if ixn not in old_label:
                                            prompt_label_list.append(ixn)

                            if prompt_label_list:
                                if break_endless_loop:
                                    break_endless_loop = False 
                                    idx += 1
                                    continue
                                print("Adding new labels for frame "  + frame_names[idx] + " = Label " + str(prompt_label_list))
                                break
                            break_endless_loop = False

                    old_label = list(set(old_label) & set(current_label)) + prompt_label_list
            idx += 1    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="sam2_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2_hiera_b+.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--maskrcnn_checkpoint",
        type=str,
        required=True,
        help="path to the MaskRCNN model checkpoint",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=['CADIS', 'CHOLECSEG8K', 'CATARACT1K'],
        help="dataset type to run the prediction on. Currently supported: CADIS, CHOLECSEG8K, CATARACT1K",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="V3",
        choices=['V1', 'V2', 'V3'],
        help=
        "V1: run SASVI prediction on the entire video with only MaskRCNN's first frame mask as input"
        "V2: run SASVI prediction on the entire video with  MaskRCNN's first frame mask as input and all gt mask as intermediate input"
        "V3: run SASVI prediction on the entire video with final SASVI implementation",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run SASVI prediction on",
    )
    parser.add_argument(
        "--gt_mask_dir",
        type=str,
        required=True,
        help="directory containing gt masks (as PNG files) from interval frames of each video",
    )
    parser.add_argument(
        "--maskrcnn_mask_dir",
        type=str,
        required=True,
        help="directory containing predicted masks from maskrcnn (as npz files) of each video",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks"
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--save_binary_mask",
        action="store_true",
        help="whether to also save per object binary masks in addition to the combined mask",
    )
    args = parser.parse_args()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    video_names = [
        p
        for p in os.listdir(args.base_video_dir)
        if os.path.isdir(os.path.join(args.base_video_dir, p))
    ]
    # adding this filter based on the dataset type
    if args.dataset_type == "CADIS":
        video_names[:] = sorted([item for item in video_names if item.startswith('train')])
    elif args.dataset_type == "CHOLECSEG8K":
        video_names[:] = sorted([item for item in video_names if item.startswith('video')])
    elif args.dataset_type == "CATARACT1K":
        video_names[:] = sorted([item for item in video_names if item.startswith('case')])
    else:        
        raise NotImplementedError

    #TODO: Use the model to predict maskrcnn mask on the fly
    maskrcnn_model, num_classes, ignore_indices, shift_by_1, palette = load_overseer(
        checkpoint_path=args.maskrcnn_checkpoint, 
        dataset=args.dataset_type)
    
    print(f"running SASVI prediction on {len(video_names)} videos:\n{video_names}")
    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        sasvi_inference(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            gt_mask_dir=args.gt_mask_dir,
            maskrcnn_mask_dir=args.maskrcnn_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            maskrcnn_model=maskrcnn_model,
            
            num_classes=num_classes,
            ignore_indices=ignore_indices,
            shift_by_1=shift_by_1,
            palette=palette,
            dataset_type=args.dataset_type,
            
            prediction_type=args.prediction_type,
            score_thresh=args.score_thresh,
            save_binary_mask=args.save_binary_mask,
        )

    print(f"completed SASVI prediction on {len(video_names)} videos -- "
          f"output masks saved to {args.output_mask_dir}"    
    )

if __name__ == "__main__":
    main()