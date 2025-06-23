# Using nnunet on top of overseer for tough instances. But not part of original paper and can be left out
#TODO: Prediction on reverse direction when overseer prediction confidence is high
#TODO: Fine-tune SAM2 on the dataset

import os
import cv2
import sys
import time
import numpy as np
import torch
import argparse
import itertools
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import find_objects
import albumentations as A
import torchvision.transforms as T
from transformers import DetrImageProcessor, DetrForSegmentation, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import get_cholecseg8k_colormap
from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import get_cataract1k_colormap

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sam2.build_sam import build_sam2_video_predictor
from src.sam2_utils import kmeans_sampling
from src.data import remap_labels, insert_component_masks
from src.utils import process_detr_outputs, process_mask2former_outputs
from src.model import get_model_instance_segmentation

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


def get_per_obj_mask(mask_path, frame_name, use_binary_mask, width, height, ignore_indices, shift_by_1, palette):
    """Split a mask into per-object masks."""
    # For CholecSeg8K, the gt mask and frame are saved according to frame rate. But overseer paths are not according to frame rate.
    if not use_binary_mask:
        mask = load_ann_png(path=os.path.join(mask_path, f"{frame_name}_rgb_mask.png"),
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
    vis_mode=False,
    save_height=299,
    save_width=299,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    per_obj_output_mask = {
        key: np.expand_dims(value, axis=0) if value.ndim == 2 else value
        for key, value in per_obj_output_mask.items()
    }
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

        if vis_mode:
            rows, cols = 4, 4  # Adjust based on the number of masks
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                if i < output_mask.shape[0]:
                    ax.imshow(output_mask[i], cmap='gray')  # Display each mask
                    ax.set_title(f"Mask {i+1}")
                ax.axis('off')
            for i in range(output_mask.shape[0], len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(os.path.join(output_mask_dir, video_name, f"{frame_name}_visualization.jpg"), bbox_inches='tight', pad_inches=0.1)
            plt.close()


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

    # set flag true if highest n labels are close together. Will opt to use nnunet for such cases
    best_key = max(selections, key=selections.get)
    best_score = selections[best_key]
    use_nnunet = False
    for key in selections:
        if key != best_key:
            if selections[key] >= best_score * 0.95:
                use_nnunet = True

    return selections, use_nnunet


class nnUNet():
    def __init__(self, model, ignore_indices):
        self.model = model
        self.ignore_indices = ignore_indices

    def get_prediction(self, image_path, reshape_size=None):
        image, props = NaturalImage2DIO().read_images([image_path])
        image = image.squeeze(1).transpose(1, 2, 0)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        image = image.transpose(2, 0, 1).reshape(3, 1, 128, 128)
        pred_mask = self.model.predict_single_npy_array(image, props, None, None, False)
        pred_mask = pred_mask.squeeze(0)

        if reshape_size is not None:
            pred_mask = cv2.resize(pred_mask, reshape_size, interpolation=cv2.INTER_NEAREST)

        object_ids = np.unique(pred_mask)
        object_ids = object_ids[object_ids >= 0].tolist()

        length = len(self.ignore_indices)
        if length > 0:
            object_ids = object_ids[:-length]
            per_obj_mask = {object_id: (pred_mask == object_id) for object_id in object_ids}
        elif length == 0:
            object_ids = object_ids[1:]
            per_obj_mask = {object_id: (pred_mask == object_id) for object_id in object_ids}
        return per_obj_mask


class MaskRCNN():
    def __init__(self, model, shift_by_1, ignore_indices, num_classes, device="cuda"):
        self.model = model
        self.transform =  T.Compose([T.ToTensor(), T.Resize((299, 299)),T.Normalize(0.0, 1.0)])
        self.shift_by_1 = shift_by_1
        self.ignore_indices = ignore_indices
        self.num_classes = num_classes
        self.device = device

    def get_prediction(self, image_list, reshape_size=None):
        images = [np.array(Image.open(i).convert('RGB')) for i in image_list]
        images = [self.transform(i).to(self.device) for i in images]
        with torch.no_grad():
            predictions = self.model(images)

        pred_mask = predictions[0]["masks"]
        pred_labels = predictions[0]["labels"]
        if self.shift_by_1: pred_labels -= 1
        remapped_pred_labels = remap_labels(pred_labels, self.num_classes, self.ignore_indices)
        binary_pred_mask = (pred_mask > 0.5).int()
        
        padded_binary_pred_mask = insert_component_masks(
            binary_pred_mask,
            remapped_pred_labels,
            self.num_classes,
            ignore_index=self.ignore_indices[0] if len(self.ignore_indices) != 0 else None)
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

        if reshape_size is not None:
            mask = np.array([resize(i, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for i in mask])
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
                
        # for cholec and cat1k, there are background class and also area with no prediction
        # for cadis, only no prediction area.  
        if self.shift_by_1:
            background_class = self.num_classes-1
            object_ids = object_ids[object_ids != background_class]
        
        per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}
        return per_obj_mask
        

class Mask2Former():
    def __init__(self, model, shift_by_1, ignore_indices, num_classes, num_train_classes, dataset_type, device="cuda"):
        self.model = model
        self.transform =  A.Compose([A.Resize(*(299, 299)), A.Normalize(0.0, 1.0)])
        self.shift_by_1 = shift_by_1
        self.ignore_indices = ignore_indices
        self.num_classes = num_classes
        self.num_train_classes = num_train_classes
        self.dataset_type = dataset_type
        self.device = device
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance", reduce_labels=True)

    def get_prediction(self, image_list, reshape_size=(299,299)):
        images = [np.array(Image.open(i)) for i in image_list]
        images = [self.transform(image=i)['image'] for i in images]
        images = [A.pytorch.functional.img_to_tensor(i).to(self.device) for i in images]
        images = torch.stack(images)
        # images = self.processor(images=[image.cpu().numpy() for image in images],
        #                         do_rescale=False,
        #                         return_tensors="pt").to(self.device)

        with torch.no_grad():
            predictions = self.model(images)
        predictions = process_mask2former_outputs(predictions,
                                           image_size=reshape_size,
                                           num_labels=self.num_train_classes,
                                           threshold=0.0) #if self.dataset_type == "CADIS" else 0.5)

        pred_mask = predictions[0]["masks"]
        pred_labels = predictions[0]["labels"]
        
        #TODO: All this dataset specific conditioning need to be more cleaner
        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            pred_labels -= 1

        # CAT1K: i think just needs to remove ignore class 0
        if self.dataset_type == "CATARACT1K":
            self.ignore_indices = []  

        if self.shift_by_1: pred_labels -= 1
        remapped_pred_labels = remap_labels(pred_labels, self.num_classes, self.ignore_indices)
        
        padded_binary_pred_mask = insert_component_masks(
            pred_mask,
            remapped_pred_labels,
            self.num_classes,
            ignore_index=self.ignore_indices[0] if len(self.ignore_indices) != 0 else None)
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

        if reshape_size is not None:
            mask = np.array([resize(i, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for i in mask])
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]        
        per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}

        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 0}

        # CADIS: need to remove last class (remaining 0 to 16) (ignore is 17. but actually 255)
        if self.dataset_type == "CADIS":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 17}
        return per_obj_mask
    

class DETR():
    def __init__(self, model, shift_by_1, ignore_indices, num_classes, num_train_classes, dataset_type, device="cuda"):
        self.model = model
        self.transform =  A.Compose([A.Resize(*(200, 200)), A.Normalize(0.0, 1.0)])
        self.shift_by_1 = shift_by_1
        self.ignore_indices = ignore_indices
        self.num_classes = num_classes
        self.num_train_classes = num_train_classes
        self.dataset_type = dataset_type
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")

    def get_prediction(self, image_list, reshape_size=(299,299)):
        images = [np.array(Image.open(i)) for i in image_list]
        images = [self.transform(image=i)['image'] for i in images]
        images = [A.pytorch.functional.img_to_tensor(i).to(self.device) for i in images]
        images = self.processor(images=[image.cpu().numpy() for image in images],
                                do_rescale=False,
                                return_tensors="pt").to(self.device)

        with torch.no_grad():
            predictions = self.model(**images)
        predictions = process_detr_outputs(predictions,
                                           image_size=reshape_size,
                                           num_labels=self.num_train_classes,
                                           threshold=0.0) #if self.dataset_type == "CADIS" else 0.5)

        pred_mask = predictions[0]["masks"]
        pred_labels = predictions[0]["labels"]
        
        #TODO: All this dataset specific conditioning need to be more cleaner
        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            pred_labels -= 1

        # CAT1K: i think just needs to remove ignore class 0
        if self.dataset_type == "CATARACT1K":
            self.ignore_indices = []  

        if self.shift_by_1: pred_labels -= 1
        remapped_pred_labels = remap_labels(pred_labels, self.num_classes, self.ignore_indices)
        
        padded_binary_pred_mask = insert_component_masks(
            pred_mask,
            remapped_pred_labels,
            self.num_classes,
            ignore_index=self.ignore_indices[0] if len(self.ignore_indices) != 0 else None)
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

        if reshape_size is not None:
            mask = np.array([resize(i, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for i in mask])
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]        
        per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}

        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 0}

        # CADIS: need to remove last class (remaining 0 to 16) (ignore is 17. but actually 255)
        if self.dataset_type == "CADIS":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 17}
        return per_obj_mask
            

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def sasvi_inference(
    predictor,
    base_video_dir,
    output_mask_dir,
    video_name,
    
    overseer_type,
    overseer_mask_dir,
    overseer_model,
    nnunet_model,

    num_classes,
    ignore_indices,
    shift_by_1,
    palette,
    dataset_type,

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

    if dataset_type == "CADIS":
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][5:]))
    else:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    per_obj_input_mask_cached = {}
    
    # load the video frames and initialize the inference state of SAM2 on this video
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    break_endless_loop = False

    # run propagation throughout the video and collect the results in a dict
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)

    idx = 0
    while idx < len(frame_names):
        # clear the prompts from previous runs
        print("New inference for segment: " + frame_names[idx])
        predictor.reset_state(inference_state=inference_state)
        buffer_length = 25

        # clean up the cached mask if it gets too big. Slightly larger than future_n_frame length
        MAX_CACHE_SIZE = 20
        while len(per_obj_input_mask_cached) > MAX_CACHE_SIZE:
            smallest_idx = min(per_obj_input_mask_cached.keys())
            del per_obj_input_mask_cached[smallest_idx]

        # this loads the masks. will add those input masks to SAM 2 inference state before propagations
        if per_obj_input_mask_cached.get(idx) is not None:
            per_obj_input_mask = per_obj_input_mask_cached.get(idx)
        else: 
            per_obj_input_mask = overseer_model.get_prediction([os.path.join(video_dir, f"{frame_names[idx]}.jpg")], reshape_size=(height, width))
            per_obj_input_mask_cached[idx] = per_obj_input_mask

        if idx > 0:
            # to make it more stable, if something ignored in overseer frame, but detected in previous sam2 frame, add it to the prompt mask
            # but didnt work for cat1k and cholec80 bcs it slowly covers the background which should have no label
            if dataset_type == "CADIS":
                per_obj_previous_mask = get_per_obj_mask(
                                            mask_path=os.path.join(output_mask_dir, video_name), 
                                            frame_name=frame_names[idx],
                                            use_binary_mask=False, 
                                            width=width, 
                                            height=height,
                                            ignore_indices=ignore_indices, 
                                            shift_by_1=shift_by_1, 
                                            palette=palette)
            
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
            if idx == 0 or (idx > 0 and object_id in yet_another_unique_label and object_id not in negative_duplicate_list):
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
        
        if len(inference_state['point_inputs_per_obj']) == 0 and len(inference_state['mask_inputs_per_obj']) == 0:
            print("Empty overseer mask, using dummy background point prompt. Happens when camera move out of scene.")
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

            if overseer_mask_dir is not None:
                save_masks_to_dir(
                    output_mask_dir=overseer_mask_dir,
                    video_name=video_name,
                    frame_name=frame_names[out_frame_idx],
                    per_obj_output_mask=per_obj_input_mask_cached[out_frame_idx],
                    height=height,
                    width=width,
                    output_palette=palette,
                    save_binary_mask=save_binary_mask,
                    num_classes=num_classes,
                    shift_by_1=shift_by_1,
                )

            future_n_frame = min(10, len(frame_names) - idx)
            per_obj_input_mask_n = []
            duplicate_list = []
            negative_duplicate_list = []
            partial_duplicate_list = []
            prompt_label_list = []

            # getting overseer prediction and caching them for future use
            for n in range(future_n_frame):
                if per_obj_input_mask_cached.get(idx+n) is not None:
                    per_obj_input_mask_n.append(per_obj_input_mask_cached.get(idx+n))
                else:
                    temp = overseer_model.get_prediction([os.path.join(video_dir, f"{frame_names[idx+n]}.jpg")], reshape_size=(height, width))
                    per_obj_input_mask_n.append(temp)
                    per_obj_input_mask_cached[idx+n] = temp

            # getting separate unique labels for each frame in per_obj_input_mask_n
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
                    #TODO: Can define this as parameter and be more dynamic
                    # also ignoring labels that appear in one frame but not in next 3 frames
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
                                    # directly append if empty, else check if the reverse pair already exist
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
                                # prompt label is the one that will actually be used for new prompting
                                # negative_duplicate_list just filters added sublist                           
                                selections, use_nnunet = choose_duplicate_label(per_obj_mask_n=per_obj_input_mask_n, duplicate_label=sublist)
                                
                                prompt_label = max(selections, key=selections.get)
                                # use nnunet almost in ensembling fashion for error propagation
                                if use_nnunet and nnunet_model is not None:
                                    per_obj_input_mask_nnunet = nnunet_model.get_prediction(os.path.join(video_dir, f"{frame_names[idx]}.jpg"), reshape_size=(width, height))
                                    area_to_check = per_obj_input_mask_n[0][prompt_label]

                                    for key in selections:
                                        if key in per_obj_input_mask_nnunet:
                                            selections[key] += np.sum(np.logical_and(area_to_check, per_obj_input_mask_nnunet[key]))
                                    prompt_label = max(selections, key=selections.get)

                                if prompt_label not in old_label:
                                    prompt_label_list.append(prompt_label)
                                negative_duplicate_list = [[x for x in slist if x != prompt_label] for slist in negative_duplicate_list]

                        # if new label without any duplicate        
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
        "--device",
        type=str,
        default="cuda",
        help="device to run the prediction on (default: cuda)",
    )
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
        "--overseer_checkpoint",
        type=str,
        required=True,
        help="path to the Overseer model checkpoint",
    )
    parser.add_argument(
        "--overseer_type",
        type=str,
        required=True,
        choices=['MaskRCNN', 'DETR', 'Mask2Former'],

        help="path to the Overseer model checkpoint",
    )
    parser.add_argument(
        "--nnunet_checkpoint",
        type=str,
        help="path to the nnUnet model checkpoint. If provided will use nnUnet prediction for tough cases.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=['CADIS', 'CHOLECSEG8K', 'CATARACT1K'],
        help="dataset type to run the prediction on. Currently supported: CADIS, CHOLECSEG8K, CATARACT1K",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run SASVI prediction on",
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
    parser.add_argument(
        "--overseer_mask_dir",
        type=str,
        help="directory to save predicted masks from overseer of each video.",
    )
    parser.add_argument(
        "--nnunet_mask_dir",
        type=str,
        help="directory to save predicted masks from nnunet of each video.",
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
    video_names = sorted(video_names)

    # adding this filter based on the dataset type
    if args.dataset_type == "CADIS":
        # video_names[:] = sorted([item for item in video_names if item.startswith('train')])
        num_classes = 18
        ignore_indices = [255]
        shift_by_1 = True
        palette = get_cadis_colormap()
        maskrcnn_hidden_ft = 32
        maskrcnn_backbone = 'ResNet18'
        
    elif args.dataset_type == "CHOLECSEG8K":
        # video_names[:] = sorted([item for item in video_names if item.startswith('video')])
        num_classes = 13
        ignore_indices = [0] if args.overseer_type == "Mask2Former" else []
        shift_by_1 = False
        palette = get_cholecseg8k_colormap()
        maskrcnn_hidden_ft = 64
        maskrcnn_backbone = 'ResNet50'

    elif args.dataset_type == "CATARACT1K":
        # video_names[:] = sorted([item for item in video_names if item.startswith('case')])
        num_classes = 14
        ignore_indices = [0] if args.overseer_type == "DETR" or args.overseer_type == "Mask2Former" else []
        shift_by_1 = False
        palette = get_cataract1k_colormap()
        maskrcnn_hidden_ft = 32
        maskrcnn_backbone = 'ResNet18'

    else:        
        raise NotImplementedError

    if args.overseer_type == "MaskRCNN":
        maskrcnn_model = get_model_instance_segmentation(
            num_classes=num_classes,
            trainable_backbone_layers=0,
            hidden_ft=maskrcnn_hidden_ft,
            custom_in_ft_box=None,
            custom_in_ft_mask=None,
            backbone=maskrcnn_backbone,
            img_size=(299, 299)
        )
        maskrcnn_model.load_state_dict(torch.load(args.overseer_checkpoint, weights_only=True, map_location='cpu'))
        maskrcnn_model = maskrcnn_model.to(args.device)
        maskrcnn_model.eval()       
        overseer_model = MaskRCNN(maskrcnn_model, shift_by_1, ignore_indices, num_classes, device=args.device)

    elif args.overseer_type == "DETR":
        num_train_classes = num_classes - len(ignore_indices) + 1
        detr_model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=100).to(args.device)
        detr_model.load_state_dict(torch.load(args.overseer_checkpoint, map_location='cpu', weights_only=True))
        detr_model.eval()
        overseer_model = DETR(detr_model, shift_by_1, ignore_indices, num_classes, num_train_classes, args.dataset_type, device=args.device)
    
    elif args.overseer_type == "Mask2Former":
        num_train_classes = num_classes - len(ignore_indices) + 1
        mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-instance",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=20).to(args.device)
        mask2former_model.load_state_dict(torch.load(args.overseer_checkpoint, map_location='cpu', weights_only=True))
        mask2former_model.eval()
        overseer_model = Mask2Former(mask2former_model, shift_by_1, ignore_indices, num_classes, num_train_classes, args.dataset_type, device=args.device)

    else: 
        raise NotImplementedError
    
    if args.nnunet_checkpoint is not None:
        nnunet_predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(args.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        nnunet_predictor.initialize_from_trained_model_folder(
            args.nnunet_checkpoint,
            use_folds=(0,),
            checkpoint_name='checkpoint_final.pth',
        )
        nnunet_model = nnUNet(nnunet_predictor, ignore_indices)
    else: nnunet_model = None

    print(f"running SASVI prediction on {len(video_names)} videos:\n{video_names}")
    start_time = time.time()
    
    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        sasvi_inference(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,

            overseer_type=args.overseer_type,
            overseer_mask_dir=args.overseer_mask_dir,
            overseer_model=overseer_model,
            nnunet_model=nnunet_model,

            num_classes=num_classes,
            ignore_indices=ignore_indices,
            shift_by_1=shift_by_1,
            palette=palette,
            dataset_type=args.dataset_type,
            
            score_thresh=args.score_thresh,
            save_binary_mask=args.save_binary_mask,
        )

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")
    print(f"completed SASVI prediction on {len(video_names)} videos -- "
          f"output masks saved to {args.output_mask_dir}"    
    )

if __name__ == "__main__":
    main()