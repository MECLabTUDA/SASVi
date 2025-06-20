# SASVi - Segment Any Surgical Video

![](figures/introduction.png#gh-light-mode-only)
![](figures/introduction_dark_mode.png#gh-dark-mode-only)

### Overview

SASVi leverages pre-trained frame-wise object detection and segmentation to re-prompt SAM2
for improved surgical video segmentation with scarcely annotated data.

![](figures/inference_scheme_v2.png#gh-light-mode-only) 
![](figures/inference_scheme_v2_dark_mode.png#gh-dark-mode-only)

### TODOs

* [ ] **The code will be refactored soon to be more modular and reusable!**
* [ ] Pre-process Cholec80 videos with out-of-body detection
* [ ] Improve SASVi by combining it with GT prompting (if available)
* [ ] Test SAM2 finetuning

### Example Results

You can find the complete segmentations of the video datasets [here](https://next.hessenbox.de/index.php/s/SmPNcMMEbBsbHB6). 

### Setup

 * Create a virtual environment of your choice and activate it.
 * Install the dependencies using `pip install -r requirements.txt`.
 * Install SDS_Playground from [here](https://github.com/MECLabTUDA/SDS_Playground)
 * Download and install SAM2 from [here](https://github.com/facebookresearch/segment-anything-2).
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

### Overseer Model Training

We provide training scripts for three different overseer models (Mask R-CNN, DETR, Mask2Former)
on three different datasets (CaDIS, CholecSeg8k, Cataract1k).

You can run the training scripts as follows:

`python train_scripts/train_<OVERSEER>_<DATASET>.py`


### SAM2 / SASVi Inference

 * SAM2: ...
 * SASVi: ...

### nnUNet Training & Inference

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

### Evaluation

 * For frame-wise segmentation evaluation:
   * `python eval_scripts/eval_<OVERSEER>_frames.py`
 * For frame-wise segmentation prediction on full videos:
   * See `python eval_scripts/eval_MaskRCNN_videos.py` for an example.
 * For video evaluation:
   1. E.g. `python eval_scripts/eval_vid_T.py --segm_root <path_to_segmentation_root> --vid_pattern 'train' --mask_pattern '*.npz' --ignore 255 --device cuda`
   2. E.g. `python eval_scripts/eval_vid_F.py --segm_root <path_to_segmentation_root> --frames_root <path_to_frames_root> --vid_pattern 'train' --frames_pattern '*.jpg' --mask_pattern '*.npz' --raft_iters 12 --device cuda`

### Citation

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
