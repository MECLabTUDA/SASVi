conda create --name sasvi python=3.10

conda activate sasvi

cd SASVi/src_sam2

pip install -e .
pip install -e ".[demo]"


/gris/gris-f/homestud/ssivakum/SASVi/src_sam2/eval_sasvi.py" --sam2_cfg sam2_hiera_l.yaml --sam2_checkpoint ./checkpoints/sam2_hiera_large.pt --base_video_dir /gris/gris-f/homestud/ssivakum/SASVi/data/video --input_mask_dir /gris/gris-f/homestud/ssivakum/SASVi/data/gt --output_mask_dir /gris/gris-f/homestud/ssivakum/SASVi/data/output
