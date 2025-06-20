#!/bin/bash

# Directory containing video files
DIR="/local/scratch/Cholec80/cholec80_full_set/videos/"
# DIR="/local/scratch/Catarakt/videos/micro/"

# Find all video files in the directory
files=$(find "$DIR" -type f -name "*.mp4" -o -name "*.mkv" -o -name "*.avi")

total_duration=0
count=0

# Loop through each file and get its duration
for file in $files; do
    duration=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    total_duration=$(echo "$total_duration + $duration" | bc)
    count=$((count + 1))
done

if [ "$count" -gt 0 ]; then
    average_duration=$(echo "$total_duration / $count" | bc -l)
    echo "Average duration (in seconds): $average_duration"
else
    echo "No video files found in the directory."
fi