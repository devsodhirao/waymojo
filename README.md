
# YOLOv3 Inference on KITTI Dataset

This repository contains code for running YOLOv3 object detection inference on the KITTI dataset and measuring performance metrics.

## Overview

- Uses PyTorch with Ultralytics YOLOv3 implementation
- Benchmarks inference time and detection performance
- Supports both RGB images and disparity maps from KITTI

## Scripts

### Python Scripts
- `scripts/yolov3_direct_inference.py`: Main inference script with timing measurements

### Shell Scripts
- `rgb_inference.sh`: Run inference on RGB images from KITTI dataset
- `run_direct_inference.sh`: Script with recursive image search capability
- `simple_inference.sh`: Simplified script for basic inference

## Performance Metrics

When running on CPU:
- Average inference time: ~0.3-0.4 seconds per image
- FPS: ~2.5-3.0 frames per second

## Requirementstorch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
pillow>=8.0.0 EOL
