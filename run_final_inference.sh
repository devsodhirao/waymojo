#!/bin/bash

# Configuration
BASE_DIR="/home/ak/kitti_dataset"
OUTPUT_DIR="output_final"
NUM_IMAGES=20

# Create output directory
mkdir -p $OUTPUT_DIR

# Add ~/.local/bin to PATH for this session
export PATH=$PATH:~/.local/bin

# Find the image_2 directory (which should contain RGB images)
IMAGE_DIR=$(find $BASE_DIR -name "image_2" -type d | head -1)

# If no image_2 directory found, use previous directory
if [ -z "$IMAGE_DIR" ]; then
    echo "No image_2 directory found, using disparity maps instead"
    IMAGE_DIR="/home/ak/kitti_dataset/testing/training/disp_occ_0"
else
    echo "Found RGB images in: $IMAGE_DIR"
fi

# Show what's in the directory
echo "Contents of image directory:"
ls -la $IMAGE_DIR | head -10

# Run inference with proper dataset
python3 scripts/yolov3_direct_inference.py \
    --image_dir $IMAGE_DIR \
    --conf_threshold 0.4 \
    --iou_threshold 0.45 \
    --img_size 640 \
    --num_images $NUM_IMAGES \
    --output_dir $OUTPUT_DIR

echo "Inference completed. Results are in $OUTPUT_DIR directory."
