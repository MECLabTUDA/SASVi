import os
import subprocess

from natsort import natsorted


# input_dir = '/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/input/'
# input_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/input/'
input_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cholec80/input/'

# output_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/_nnUNet_prediction/'
# output_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/_nnUNet_prediction/'
output_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cholec80/_nnUNet_prediction/'

# pattern = 'train'
pattern = None

# d = 'Dataset667_CaDISv2EXP2'
# d = 'Dataset888_Cataracts1kSegm'
# d = 'Dataset777_CholecSeg8k'

dirs = natsorted(os.listdir(input_dir))
for dir in dirs:

    full_dir = os.path.join(input_dir, dir)
    if not os.path.isdir(full_dir) or (pattern is not None and pattern not in dir):
        continue

    print(f'Processing directory: {full_dir}')

    output_path = os.path.join(output_base_dir, dir)
    os.makedirs(output_path, exist_ok=True)

    # Run nnUNet inference
    command = [
        'nnUNetv2_predict',
        '-d', d,
        '-i', full_dir,
        '-o', output_path,
        # '-f', '0', '1', '2', '3', '4',
        '-f', '0',
        '-tr', 'nnUNetTrainer_400epochs',
        '-c', '2d',
        '-p', 'nnUNetResEncUNetMPlans',
        '-npp', '8',
        '-nps', '8'
    ]
    subprocess.run(command)
