# ===== test_yolov3.mojo =====
from tensor import Tensor, TensorShape
from time import now
from collections import List

alias DT = DType.float32

# Import model
import yolov3_model

fn test_yolov3_model() raises:
    """Test the complete YOLOv3 model."""
    print("Testing YOLOv3 model...")
    
    # Create model
    var model = yolov3_model.YOLOv3(80)  # COCO dataset has 80 classes
    
    # Create dummy input
    var input = Tensor[DT](TensorShape(1, 3, 416, 416))
    
    # Initialize with dummy data
    for i in range(input.num_elements()):
        input[i] = Float32((i * 3) % 256) / 255.0
    
    # Test forward pass
    print("Running forward pass...")
    var start_time = now()
    var predictions = model.forward(input, training=False)
    var end_time = now()
    
    print("Forward pass completed in:", (end_time - start_time) / 1e6, "ms")
    print("Number of prediction scales:", len(predictions))
    
    for i in range(len(predictions)):
        print("Scale", i, "prediction shape:", predictions[i].shape())
    
    # Test prediction pipeline
    print("\nTesting prediction pipeline...")
    var pred_start_time = now()
    var detections = model.predict(input, 0.5)
    var pred_end_time = now()
    
    print("Prediction completed in:", (pred_end_time - pred_start_time) / 1e6, "ms")
    print("Number of detections:", len(detections))

fn test_individual_components() raises:
    """Test individual model components."""
    print("\nTesting individual components...")
    
    # Test Darknet-53 backbone
    print("Testing Darknet-53 backbone...")
    var backbone = yolov3_model.Darknet53()
    var dummy_input = Tensor[DT](TensorShape(1, 3, 416, 416))
    
    # Initialize with dummy data
    for i in range(dummy_input.num_elements()):
        dummy_input[i] = Float32((i * 5) % 256) / 255.0
    
    var features = backbone.forward(dummy_input, training=False)
    print("Number of feature scales:", len(features))
    for i in range(len(features)):
        print("Feature scale", i, "shape:", features[i].shape())
    
    # Test YOLOv3 head
    print("\nTesting YOLOv3 head...")
    var head = yolov3_model.YOLOv3Head(80)
    var head_predictions = head.forward(features, training=False)
    print("Number of head predictions:", len(head_predictions))
    for i in range(len(head_predictions)):
        print("Head prediction", i, "shape:", head_predictions[i].shape())

fn benchmark_model() raises:
    """Benchmark the YOLOv3 model."""
    print("\nBenchmarking YOLOv3 model...")
    
    var model = yolov3_model.YOLOv3(80)
    var input = Tensor[DT](TensorShape(1, 3, 416, 416))
    
    # Initialize with random data
    for i in range(input.num_elements()):
        input[i] = Float32((i * 7) % 256) / 255.0
    
    # Warmup runs
    print("Warmup runs...")
    for i in range(5):
        _ = model.forward(input, training=False)
    
    # Benchmark runs
    print("Benchmark runs...")
    var num_runs = 10
    var total_time: Float64 = 0.0
    
    for i in range(num_runs):
        var start_time = now()
        _ = model.forward(input, training=False)
        var end_time = now()
        total_time += Float64(end_time - start_time)
    
    var avg_time = total_time / Float64(num_runs * 1e6)  # Convert to ms
    var fps = 1000.0 / avg_time
    
    print("Average inference time:", avg_time, "ms")
    print("Average FPS:", fps)

fn main():
    print("YOLOv3 Model Testing")
    print("===================\n")
    
    try:
        # Test complete model
        test_yolov3_model()
        
        # Test individual components
        test_individual_components()
        
        # Benchmark performance
        benchmark_model()
        
        print("\n✅ All YOLOv3 tests completed!")
    except e:
        print("❌ Error:", e)