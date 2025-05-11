import os
import time
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 inference on KITTI dataset using PyTorch')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory with images')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to process')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

def find_images_recursive(directory):
    """
    Recursively find all image files in a directory and its subdirectories.
    """
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob(os.path.join(directory, '**', ext), recursive=True))
    
    return sorted(image_files)

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
        from ultralytics import YOLO
        model = YOLO('yolov3.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find image files recursively in the directory
    print(f"Searching for images in {args.image_dir}...")
    image_files = find_images_recursive(args.image_dir)
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.image_dir}")
        # Try direct listing
        if os.path.exists(args.image_dir):
            print("Directory exists but no images found. Contents:")
            os.system(f"ls -la {args.image_dir}")
        else:
            print(f"Directory {args.image_dir} does not exist!")
        return
    
    num_images = min(len(image_files), args.num_images)
    print(f"Found {len(image_files)} images. Processing {num_images} images...")
    
    # Process images
    total_inference_time = 0
    total_detections = 0
    
    for i in range(num_images):
        image_path = image_files[i]
        print(f"Processing image {i+1}/{num_images}: {os.path.basename(image_path)}")
        
        # Start timing
        start_time = time.time()
        
        # Run inference
        try:
            results = model(image_path)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Count detections
            num_detections = len(results[0].boxes)
            total_detections += num_detections
            
            # Save results
            result_image = results[0].plot()
            output_filename = f"result_{os.path.basename(image_path)}"
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, result_image)
            
            # Print detection info
            print(f"  Detected {num_detections} objects in {inference_time:.4f} seconds")
            
            # Print detection details
            if num_detections > 0:
                boxes = results[0].boxes
                for j in range(num_detections):
                    confidence = float(boxes.conf[j])
                    class_id = int(boxes.cls[j])
                    class_name = results[0].names[class_id]
                    print(f"    - {class_name}: {confidence:.4f}")
            
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
