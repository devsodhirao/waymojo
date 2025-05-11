import os
import time
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 inference on KITTI dataset using PyTorch')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to KITTI dataset')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to process')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

def find_image_directory(dataset_path):
    """
    Find the directory containing images in the KITTI dataset.
    """
    # Check possible image directories
    possible_dirs = [
        os.path.join(dataset_path, 'testing', 'image_2'),
        os.path.join(dataset_path, 'training', 'image_2'),
        os.path.join(dataset_path, 'image_2'),
        os.path.join(dataset_path, 'images'),
        dataset_path
    ]
    
    for directory in possible_dirs:
        if os.path.exists(directory):
            # Check if directory contains images
            image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                print(f"Found image directory: {directory}")
                return directory
    
    print("Error: Could not find image directory in the dataset")
    return None

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load YOLOv3 model from Ultralytics
    print("Loading YOLOv3 model...")
    try:
        model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
        model.conf = args.conf_threshold  # Set confidence threshold
        model.iou = args.iou_threshold    # Set IoU threshold
        model.to(device)                  # Move model to device
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with trusted_host...")
        # Try with trusted host flag
        os.system('pip install -q torch ultralytics')
        try:
            import ultralytics
            from ultralytics import YOLO
            model = YOLO('yolov3.pt')
            print("Model loaded using YOLO method!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return
    
    # Find the directory containing images
    dataset_images_dir = find_image_directory(args.dataset_path)
    if not dataset_images_dir:
        return
    
    image_files = sorted([f for f in os.listdir(dataset_images_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print(f"Error: No images found in {dataset_images_dir}")
        return
    
    num_images = min(len(image_files), args.num_images)
    print(f"Found {len(image_files)} images. Processing {num_images} images...")
    
    # Process images
    total_inference_time = 0
    total_detections = 0
    
    for i in range(num_images):
        image_path = os.path.join(dataset_images_dir, image_files[i])
        print(f"Processing image {i+1}/{num_images}: {image_files[i]}")
        
        # Load image
        img = Image.open(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        # Start timing
        start_time = time.time()
        
        # Run inference
        try:
            results = model(img)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Count detections
            if hasattr(results, 'xyxy'):
                # Old model API
                num_detections = len(results.xyxy[0])
            else:
                # New model API
                num_detections = len(results[0].boxes)
            
            total_detections += num_detections
            
            # Save results
            output_path = os.path.join(args.output_dir, f"result_{image_files[i]}")
            results.save(save_dir=args.output_dir)
            
            # Print detection info
            print(f"  Detected {num_detections} objects in {inference_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
    
    # Print summary
    if num_images > 0:
        avg_inference_time = total_inference_time / num_images
        print("\n=== Summary ===")
        print(f"Total images processed: {num_images}")
        print(f"Total objects detected: {total_detections}")
        print(f"Total inference time: {total_inference_time:.4f} seconds")
        print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
        print(f"Average FPS: {1/avg_inference_time:.2f}")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
