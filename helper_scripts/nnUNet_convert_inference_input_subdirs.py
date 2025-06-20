import os

from PIL import Image
from tqdm import tqdm
from natsort import natsorted

# input_dir = '/local/scratch/CATARACTS-videos-processed/'
# input_dir = '/local/scratch/Cataract1kSegmentation-frames-SegmSubset-15FPS/'
input_dir = '/local/scratch/Cholec80-frames-CholecSeg8kSubset-12.5FPS/'

# output_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/input/'
# output_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/input/'
output_base_dir = '/home/yfrisch_locale/DATA/SASVi_full_Cholec80/input/'

# pattern = 'train'
pattern = None
target_size = (128, 128)


dirs = natsorted(os.listdir(input_dir))
for dir in dirs:

    full_dir = os.path.join(input_dir, dir)
    if not os.path.isdir(full_dir) or (pattern is not None and pattern not in dir):
        continue

    output_path = os.path.join(output_base_dir, dir)
    os.makedirs(output_path, exist_ok=True)

    # Process each .jpg file in the full_dir
    for file in tqdm(os.listdir(full_dir), desc=f'Processing {full_dir}'):
        if file.endswith('.jpg'):
            jpg_file_path = os.path.join(full_dir, file)
            png_file_path = os.path.join(output_path, file[:-4] + '_0000.png')  # Change extension to .png

            # Open the .jpg file and convert to .png
            with Image.open(jpg_file_path) as img:
                resized_img = img.resize(target_size, Image.ANTIALIAS)
                resized_img.save(png_file_path, 'PNG')
                # print(png_file_path)