#!/bin/bash

# Configuration
DATASET_PATH="/home/ak/kitti_dataset/testing"  # Path to the testing dataset
OUTPUT_DIR="output"
NUM_IMAGES=20  # Process 20 images

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Install dependencies
pip install torch ultralytics opencv-python pillow

# Add ~/.local/bin to PATH for this session
export PATH=$PATH:~/.local/bin

# Run inference
python3 scripts/yolov3_pytorch_inference.py \
    --dataset_path $DATASET_PATH \
    --conf_threshold 0.5 \
    --iou_threshold 0.45 \
    --img_size 640 \
    --num_images $NUM_IMAGES \
    --output_dir $OUTPUT_DIR

echo "Inference completed. Results are in $OUTPUT_DIR directory."
