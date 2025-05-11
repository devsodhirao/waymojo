import os
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Command line arguments
parser = argparse.ArgumentParser(description='YOLOv3 inference on KITTI dataset')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to KITTI dataset')
parser.add_argument('--weights_path', type=str, required=True, help='Path to YOLOv3 weights')
parser.add_argument('--config_path', type=str, required=True, help='Path to YOLOv3 config file')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS threshold')
parser.add_argument('--img_size', type=int, default=416, help='Input image size')
parser.add_argument('--num_images', type=int, default=10, help='Number of images to process')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# KITTI class names
classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]

# Colors for visualization
colors = [
    (0, 0, 255),    # Red for Car
    (0, 165, 255),  # Orange for Van
    (0, 255, 255),  # Yellow for Truck
    (0, 255, 0),    # Green for Pedestrian
    (255, 0, 0),    # Blue for Person_sitting
    (255, 0, 255),  # Purple for Cyclist
    (255, 255, 0),  # Cyan for Tram
    (128, 128, 128) # Gray for Misc
]

def load_model(weights_path, config_path, conf_threshold, nms_threshold):
    """
    Load YOLOv3 model with given weights and configuration.
    """
    print(f"Loading YOLOv3 model from {weights_path}...")
    
    # Load network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Set backend (use CUDA if available)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA is available! Using GPU...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("CUDA not available. Using CPU...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.5.4+
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # Older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    print("Model loaded successfully!")
    return net, output_layers

def prepare_image(img_path, img_size):
    """
    Prepare image for inference.
    """
    # Read and resize image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read image {img_path}")
        return None, None, None
    
    height, width, channels = image.shape
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (img_size, img_size), 
                                [0, 0, 0], True, crop=False)
    
    return image, blob, (width, height)

def draw_predictions(img, class_ids, confidences, boxes, img_width, img_height):
    """
    Draw bounding boxes and labels on the image.
    """
    img_copy = img.copy()
    
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        # Draw bounding box
        color = colors[class_ids[i] % len(colors)]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img_copy, (x, y - 20), (x + len(label) * 8, y), color, -1)
        cv2.putText(img_copy, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_copy

def process_image(image_path, net, output_layers, conf_threshold, nms_threshold, img_size):
    """
    Process a single image for inference.
    """
    # Prepare image
    image, blob, dimensions = prepare_image(image_path, img_size)
    if image is None:
        return None, None, 0
    
    img_width, img_height = dimensions
    
    # Start timing
    start_time = time.time()
    
    # Run forward pass
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # Object detected
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                w = int(detection[2] * img_width)
                h = int(detection[3] * img_height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Filter detections based on NMS results
    filtered_class_ids = []
    filtered_confidences = []
    filtered_boxes = []
    
    if len(indices) > 0:
        try:
            # OpenCV 4.5.4+
            for i in indices:
                idx = i
                filtered_class_ids.append(class_ids[idx])
                filtered_confidences.append(confidences[idx])
                filtered_boxes.append(boxes[idx])
        except:
            # Older OpenCV versions
            for i in indices:
                idx = i[0]
                filtered_class_ids.append(class_ids[idx])
                filtered_confidences.append(confidences[idx])
                filtered_boxes.append(boxes[idx])
    
    return image, (filtered_class_ids, filtered_confidences, filtered_boxes), inference_time

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
    # Load YOLOv3 model
    net, output_layers = load_model(
        args.weights_path, 
        args.config_path, 
        args.conf_threshold, 
        args.nms_threshold
    )
    
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
        
        # Run inference
        image, detections, inference_time = process_image(
            image_path, 
            net, 
            output_layers, 
            args.conf_threshold, 
            args.nms_threshold, 
            args.img_size
        )
        
        if image is None:
            continue
        
        # Accumulate statistics
        class_ids, confidences, boxes = detections
        total_inference_time += inference_time
        total_detections += len(class_ids)
        
        # Draw predictions
        result_image = draw_predictions(
            image, 
            class_ids, 
            confidences, 
            boxes, 
            image.shape[1], 
            image.shape[0]
        )
        
        # Save result
        output_path = os.path.join(args.output_dir, f"result_{image_files[i]}")
        cv2.imwrite(output_path, result_image)
        
        # Print detections for this image
        print(f"  Detected {len(class_ids)} objects in {inference_time:.4f} seconds")
        for j in range(len(class_ids)):
            if j < len(class_ids) and j < len(confidences) and class_ids[j] < len(classes):
                print(f"    - {classes[class_ids[j]]}: {confidences[j]:.4f}")
    
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
