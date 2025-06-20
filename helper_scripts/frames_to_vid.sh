#!/bin/bash

# Define the image folder and output video file name

# IMAGE_FOLDER="/local/scratch/Cataract1kSegmentation-frames-SegmSubset-15FPS/case_5014/"
# IMAGE_FOLDER="/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/nnUNet_prediction_pp_rgb_overlay/case_5014"
# IMAGE_FOLDER="/home/yfrisch_locale/DATA/SASVi_full_Cataract1k/sasvi_prediction_rgb_overlay/case_5014"

# IMAGE_FOLDER="/local/scratch/CATARACTS-videos-processed/train23"
# IMAGE_FOLDER="/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/sasvi_prediction_rgb_overlay/train23"
IMAGE_FOLDER="/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/nnUNet_prediction_rgb_overlay/train23"

# IMAGE_FOLDER="/local/scratch/Cholec80-frames-CholecSeg8kSubset-12.5FPS/video55"
# IMAGE_FOLDER="/home/yfrisch_locale/DATA/SASVi_full_Cholec80/sasvi_prediction_rgb_overlay/video55"
# IMAGE_FOLDER="/home/yfrisch_locale/DATA/SASVi_full_Cholec80/nnUNet_prediction_rgb_overlay/video55"

OUTPUT_VIDEO="/home/yfrisch_locale/DATA/SASVi_full_CATARACTS/train23_nnunet.mp4"
TEMP_FILE="images.txt"


# Create a file with the list of images in the correct format for concat
ls "$IMAGE_FOLDER"/*.jpg | natsort | sed "s|^|file |; s|$| |g" > "$TEMP_FILE"

# Create the video from the images
ffmpeg -f concat -safe 0 -i "$TEMP_FILE" -framerate 1 -c:v libx264 -pix_fmt yuv420p "$OUTPUT_VIDEO"

# Clean up
rm "$TEMP_FILE"

echo "Video created: $OUTPUT_VIDEO"