#!/bin/bash

# Define source and target directories
# SOURCE_FOLDER="/local/scratch/Catarakt/videos/micro"
# SOURCE_FOLDER="/local/scratch/Cholec80/cholec80_full_set/videos"
SOURCE_FOLDER="/local/scratch/Cataract-1K/Segmentation_dataset/videos/"

# TARGET_FOLDER="/local/scratch/CATARACTS-frames-15FPS"  # 30 FPS original framerate
# TARGET_FOLDER="/local/scratch/Cholec80-frames-12.5FPS"  # 25 FPS original framerate
TARGET_FOLDER="/local/scratch/Cataract1kSegmentation-frames-SegmSubset-15FPS"  # 60 FPS original framerate

# Create the target folder if it doesn't exist
mkdir -p "$TARGET_FOLDER"

# Loop through all .mp4 files in the source folder
for video in "$SOURCE_FOLDER"/*.mp4; do

  # Extract the video filename without the path and extension
  video_filename=$(basename "$video" .mp4)

  # Create a sub-folder inside the target folder named after the video file
  video_target_folder="$TARGET_FOLDER/$video_filename"
  mkdir -p "$video_target_folder"

  # Extract every n-th frame and save them into the sub-folder
  ffmpeg -i "$video" -vf "select=not(mod(n\,2))" -vsync vfr -q:v 5 -start_number 0 "$video_target_folder/%010d.jpg"

  echo "Processed $video and saved frames to $video_target_folder"
done
