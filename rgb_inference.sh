#!/bin/bash

# Configuration
IMAGE_DIR="/home/ak/kitti_dataset/testing/training/image_2"
OUTPUT_DIR="rgb_output"
NUM_IMAGES=20

# Create output directory
mkdir -p $OUTPUT_DIR

# Show the first few images
echo "First few RGB images:"
ls -la $IMAGE_DIR | head -10

# Run inference on RGB images
python3 scripts/yolov3_direct_inference.py \
    --image_dir "$IMAGE_DIR" \
    --conf_threshold 0.4 \
    --iou_threshold 0.45 \
    --img_size 640 \
    --num_images $NUM_IMAGES \
    --output_dir $OUTPUT_DIR

echo "Inference complete. Results saved to $OUTPUT_DIR"
