#!/bin/bash

# Configuration - use recursive search to find images
IMAGE_DIR="/home/ak/kitti_dataset/testing"
OUTPUT_DIR="output"
NUM_IMAGES=20  # Process 20 images

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Add ~/.local/bin to PATH for this session
export PATH=$PATH:~/.local/bin

# Print the contents of the testing directory to debug
echo "Contents of testing directory:"
ls -la $IMAGE_DIR

# Look for specific directories
echo "Looking for image_2 directory:"
find $IMAGE_DIR -name "image_2" -type d

# Look for image files
echo "Looking for image files:"
find $IMAGE_DIR -name "*.png" | head -5

# Run inference
python3 scripts/yolov3_direct_inference.py \
    --image_dir $IMAGE_DIR \
    --conf_threshold 0.5 \
    --iou_threshold 0.45 \
    --img_size 640 \
    --num_images $NUM_IMAGES \
    --output_dir $OUTPUT_DIR

echo "Inference completed. Results are in $OUTPUT_DIR directory."
