#!/bin/bash

echo "===== Disparity Map Results ====="
ls -la ~/yolo_kitti_project/output | wc -l
echo "Number of result files"

echo "===== RGB Image Results ====="
ls -la ~/yolo_kitti_project/rgb_output | wc -l
echo "Number of result files"

echo "===== Sample Images ====="
echo "First 3 disparity map results:"
ls -la ~/yolo_kitti_project/output | head -3

echo "First 3 RGB image results:"
ls -la ~/yolo_kitti_project/rgb_output | head -3
