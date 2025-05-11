# ===== yolo_utils.mojo =====
from tensor import Tensor, TensorShape
from math import sqrt
from builtin.math import min, max
from collections import List

alias DT = DType.float32

# Helper function for max with Float32 
fn max_float(a: Float32, b: Float32) -> Float32:
    return a if a > b else b

# Helper function for min with Float32
fn min_float(a: Float32, b: Float32) -> Float32:
    return a if a < b else b

fn preprocess_image(width: Int, height: Int, channels: Int = 3, target_size: Int = 416) -> Tensor[DT]:
    """
    Preprocess an image for YOLOv3 inference.
    For now, creates a placeholder tensor - actual image loading would use MAX Engine.
    """
    var tensor = Tensor[DT](TensorShape(1, channels, target_size, target_size))
    
    # Placeholder: fill with mock data
    for i in range(tensor.num_elements()):
        tensor[i] = Float32((i * 3) % 256) / 255.0
    
    return tensor

fn xywh_to_xyxy(boxes: Tensor[DT]) -> Tensor[DT]:
    """Convert boxes from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)."""
    var output = Tensor[DT](boxes.shape())
    
    for i in range(boxes.shape()[0]):
        var x_center = boxes[i * 4 + 0]
        var y_center = boxes[i * 4 + 1]
        var width = boxes[i * 4 + 2]
        var height = boxes[i * 4 + 3]
        
        output[i * 4 + 0] = x_center - width / 2   # x_min
        output[i * 4 + 1] = y_center - height / 2  # y_min
        output[i * 4 + 2] = x_center + width / 2   # x_max
        output[i * 4 + 3] = y_center + height / 2  # y_max
    
    return output

fn xyxy_to_xywh(boxes: Tensor[DT]) -> Tensor[DT]:
    """Convert boxes from (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height)."""
    var output = Tensor[DT](boxes.shape())
    
    for i in range(boxes.shape()[0]):
        var x_min = boxes[i * 4 + 0]
        var y_min = boxes[i * 4 + 1]
        var x_max = boxes[i * 4 + 2]
        var y_max = boxes[i * 4 + 3]
        
        output[i * 4 + 0] = (x_min + x_max) / 2   # x_center
        output[i * 4 + 1] = (y_min + y_max) / 2   # y_center
        output[i * 4 + 2] = x_max - x_min         # width
        output[i * 4 + 3] = y_max - y_min         # height
    
    return output

fn calculate_iou(box1: Tensor[DT], box2: Tensor[DT]) -> Float32:
    """Calculate IoU between two boxes in xyxy format."""
    # Get coordinates
    var x1_min = box1[0]
    var y1_min = box1[1]
    var x1_max = box1[2]
    var y1_max = box1[3]
    
    var x2_min = box2[0]
    var y2_min = box2[1]
    var x2_max = box2[2]
    var y2_max = box2[3]
    
    # Calculate intersection
    var inter_x_min = max_float(x1_min, x2_min)
    var inter_y_min = max_float(y1_min, y2_min)
    var inter_x_max = min_float(x1_max, x2_max)
    var inter_y_max = min_float(y1_max, y2_max)
    
    var inter_width = max_float(0.0, inter_x_max - inter_x_min)
    var inter_height = max_float(0.0, inter_y_max - inter_y_min)
    var intersection = inter_width * inter_height
    
    # Calculate union
    var area1 = (x1_max - x1_min) * (y1_max - y1_min)
    var area2 = (x2_max - x2_min) * (y2_max - y2_min)
    var union = area1 + area2 - intersection
    
    if union == 0.0:
        return 0.0
    return intersection / union

fn non_max_suppression(
    boxes: Tensor[DT],
    scores: Tensor[DT],
    iou_threshold: Float32 = 0.5,
    score_threshold: Float32 = 0.5
) -> List[Int]:
    """Perform non-maximum suppression."""
    var keep_indices = List[Int]()
    var num_boxes = boxes.shape()[0]
    
    # Simple selection with score threshold
    for i in range(num_boxes):
        if scores[i] >= score_threshold:
            var keep = True
            
            # Check against already kept boxes
            for j in range(len(keep_indices)):
                var kept_idx = keep_indices[j]
                var box1 = Tensor[DT](TensorShape(4))
                var box2 = Tensor[DT](TensorShape(4))
                
                # Extract boxes
                for k in range(4):
                    box1[k] = boxes[i * 4 + k]
                    box2[k] = boxes[kept_idx * 4 + k]
                
                if calculate_iou(box1, box2) > iou_threshold:
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
    
    return keep_indices

# ===== Anchor utilities =====

fn get_anchors(scale_idx: Int) -> List[Float32]:
    """Get anchor boxes for a specific scale."""
    if scale_idx == 0:
        return List[Float32](10, 13, 16, 30, 33, 23)      # Small objects (52x52)
    elif scale_idx == 1:
        return List[Float32](30, 61, 62, 45, 59, 119)     # Medium objects (26x26)
    else:
        return List[Float32](116, 90, 156, 198, 373, 326) # Large objects (13x13)

fn scale_anchors(anchors: List[Float32], input_size: Int, grid_size: Int) -> List[Float32]:
    """Scale anchor boxes for specific grid size."""
    var stride = input_size // grid_size
    var scaled_anchors = List[Float32]()
    
    for i in range(len(anchors)):
        scaled_anchors.append(anchors[i] / stride)
    
    return scaled_anchors

# ===== Testing =====

fn test_utils() raises:
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test image preprocessing
    var img_tensor = preprocess_image(640, 480, 3, 416)
    print("Preprocessed image shape:", img_tensor.shape())
    
    # Test box conversions
    var boxes_xywh = Tensor[DT](TensorShape(2, 4))
    boxes_xywh[0] = 100.0  # x_center
    boxes_xywh[1] = 100.0  # y_center
    boxes_xywh[2] = 50.0   # width
    boxes_xywh[3] = 50.0   # height
    
    var boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    print("Converted box:", boxes_xyxy[0], boxes_xyxy[1], boxes_xyxy[2], boxes_xyxy[3])
    
    # Test back conversion
    var boxes_xywh_again = xyxy_to_xywh(boxes_xyxy)
    print("Back converted:", boxes_xywh_again[0], boxes_xywh_again[1], boxes_xywh_again[2], boxes_xywh_again[3])
    
    # Test IoU calculation
    var box1 = Tensor[DT](TensorShape(4))
    var box2 = Tensor[DT](TensorShape(4))
    box1[0] = 0.0; box1[1] = 0.0; box1[2] = 10.0; box1[3] = 10.0
    box2[0] = 5.0; box2[1] = 5.0; box2[2] = 15.0; box2[3] = 15.0
    
    var iou_val = calculate_iou(box1, box2)
    print("IoU between overlapping boxes:", iou_val)
    
    # Test anchor utilities
    var anchors = get_anchors(0)
    print("Anchors for scale 0:", len(anchors), "values")
    var scaled = scale_anchors(anchors, 416, 52)
    print("Scaled anchors:", len(scaled), "values")