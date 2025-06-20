import os
import subprocess

from natsort import natsorted


# predictions_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/_nnUNet_prediction/'
# predictions_dir = '/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/_nnUNet_prediction/'
predictions_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cholec80/_nnUNet_prediction/'

# pp_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/nnUNet_prediction_pp/'
# pp_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/nnUNet_prediction_pp/'
pp_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cholec80/nnUNet_prediction_pp/'

# d = 'Dataset888_Cataracts1kSegm'
# d = 'Dataset667_CaDISv2EXP2'
d = 'Dataset777_CholecSeg8k'

# pattern = 'train'
pattern = None

dirs = natsorted(os.listdir(predictions_dir))
for dir in dirs:

    full_dir = os.path.join(predictions_dir, dir)
    if not os.path.isdir(full_dir) or (pattern is not None and pattern not in dir):
        continue

    print(f'Processing directory: {full_dir}')

    output_path = os.path.join(pp_base_dir, dir)
    os.makedirs(output_path, exist_ok=True)

    # Run nnUNet inference
    command = [
        'nnUNetv2_apply_postprocessing',
        '-i', full_dir,
        '-o', output_path,
        '-pp_pkl_file', f'/home/yfrisch_locale/nnUNet/nnUNet_results/{d}/nnUNetTrainer_400epochs__nnUNetResEncUNetMPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl',
        '-np', '8',
        '-plans_json', f'/home/yfrisch_locale/nnUNet/nnUNet_results/{d}/nnUNetTrainer_400epochs__nnUNetResEncUNetMPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json',
    ]
    subprocess.run(command)
