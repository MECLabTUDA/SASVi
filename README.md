<div id="top" align="center">

# SASVi - Segment Any Surgical Video (IPCAI 2025)

  [![arXiv](https://img.shields.io/badge/arXiv-2502.09653-b31b1b.svg)](https://arxiv.org/abs/2502.09653)
  [![Paper](https://img.shields.io/badge/Paper-Visit-blue)](https://link.springer.com/article/10.1007/s11548-025-03408-y)
  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/SsharvienKumar/SASVi)

</div>

![](figures/introduction.png#gh-light-mode-only)
![](figures/introduction_dark_mode.png#gh-dark-mode-only)


## Overview

SASVi leverages pre-trained frame-wise object detection and segmentation to re-prompt SAM2
for improved surgical video segmentation with scarcely annotated data.

![](figures/inference_scheme_v2.png#gh-light-mode-only) 
![](figures/inference_scheme_v2_dark_mode.png#gh-dark-mode-only)


## Example Results

* You can find the complete segmentations of the video datasets [here](https://huggingface.co/SsharvienKumar/SASVi/tree/main/dataset).
* Checkpoints of the all the overseers can be found [here](https://huggingface.co/SsharvienKumar/SASVi/tree/main/checkpoints).


## Setup

 * Create a virtual environment of your choice and activate it: `conda create -n sasvi python=3.11 && conda activate sasvi`
 * Install `torch>=2.3.1` and `torchvision>=0.18.1` following the instructions from [here](https://pytorch.org/get-started/locally/)
 * Install the dependencies using `pip install -r requirements.txt`
 * Install SDS_Playground from [here](https://github.com/MECLabTUDA/SDS_Playground)
 * Install SAM2 using `cd src/sam2 && pip install -e .`
 * Place SAM2 [checkpoints](https://github.com/facebookresearch/sam2/tree/main#model-description) at `src/sam2/checkpoints`
 * Convert video files to frame folders using `bash helper_scripts/video_to_frames.sh`. The output should be in the format:
   ```
   <video_root>
   ├── <video1>
   │   ├── 0001.jpg
   │   ├── 0002.jpg
   │   └── ...
   ├── <video2>
   │   ├── 0001.jpg
   │   ├── 0002.jpg
   │   └── ...
   └── ...
   ```


## Overseer Model Training

We provide training scripts for three different overseer models (Mask R-CNN, DETR, Mask2Former)
on three different datasets (CaDIS, CholecSeg8k, Cataract1k).

You can run the training scripts as follows:

`python train_scripts/train_<OVERSEER>_<DATASET>.py`


## SASVi Inference

The frames in the video needs to be extracted beforehand and placed in the formatting above. More optional arguments can be found in the script directly.

```
cd src/sam2 && python eval_sasvi.py \
--sam2_cfg              configs/sam2.1_hiera_l.yaml \
--sam2_checkpoint       ./checkpoints/<SAM2_CHECKPOINT>.pt \
--overseer_checkpoint   <PATH_TO_OVERSEER_CHECKPOINT>.pth \
--overseer_type         <NAME_OF_OVERSEER> \
--dataset_type          <NAME_OF_DATASET> \
--base_video_dir        <PATH_TO_VIDEO_ROOT> \
--output_mask_dir       <OUTPUT_PATH_TO_SASVi_MASK> \
--overseer_mask_dir     <OPTIONAL - OUTPUT_PATH_TO_OVERSEER_MASK>
```


## nnUNet Training & Inference

Fold 0: `nnUNetv2_train DATASET_ID 2d 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_400epochs --npz`

Fold 1: `nnUNetv2_train DATASET_ID 2d 1 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_400epochs --npz`

Fold 2: `nnUNetv2_train DATASET_ID 2d 2 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_400epochs --npz`

Fold 3: `nnUNetv2_train DATASET_ID 2d 3 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_400epochs --npz`

Fold 4: `nnUNetv2_train DATASET_ID 2d 4 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_400epochs --npz`


Then find the best configuration using 

`nnUNetv2_find_best_configuration DATASET_ID -c 2d -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_400epochs`

And run inference using 

`nnUNetv2_predict -d DATASET_ID -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer_400epochs -c 2d -p nnUNetResEncUNetMPlans`

Once inference is completed, run postprocessing 

`nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file .../postprocessing.pkl -np 8 -plans_json .../plans.json`


## Evaluation

 * For frame-wise segmentation evaluation:
   * `python eval_scripts/eval_<OVERSEER>_frames.py`
 * For frame-wise segmentation prediction on full videos:
   * See `python eval_scripts/eval_MaskRCNN_videos.py` for an example.
 * For video evaluation:
   1. E.g. `python eval_scripts/eval_vid_T.py --segm_root <path_to_segmentation_root> --vid_pattern 'train' --mask_pattern '*.npz' --ignore 255 --device cuda`
   2. E.g. `python eval_scripts/eval_vid_F.py --segm_root <path_to_segmentation_root> --frames_root <path_to_frames_root> --vid_pattern 'train' --frames_pattern '*.jpg' --mask_pattern '*.npz' --raft_iters 12 --device cuda`


## TODOs

* [ ] **The code will be refactored soon to be more modular and reusable!**
* [ ] Pre-process Cholec80 videos with out-of-body detection
* [ ] Improve SASVi by combining it with GT prompting (if available)
* [ ] Test SAM2 finetuning


## Citation

If you use SASVi in your research, please cite our paper:

```
@article{sivakumar2025sasvi,
  title={SASVi: segment any surgical video},
  author={Sivakumar, Ssharvien Kumar and Frisch, Yannik and Ranem, Amin and Mukhopadhyay, Anirban},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--11},
  year={2025},
  publisher={Springer}
}
```
