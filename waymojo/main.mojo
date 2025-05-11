# ===== main.mojo =====
from tensor import Tensor, TensorSpec
from algorithm import parallelize, vectorize
from math import sqrt, exp
import math_utils  # Use our custom math utilities instead of builtin and time

# Import our modules (proper Mojo import syntax)
import matrix_ops
import conv_layers
import yolo_utils

fn main():
    print("YOLOv3 Implementation in Mojo")
    print("============================\n")
    
    # Run component tests
    try:
        run_component_tests()
        print("\n✅ All tests passed!")
    except e:
        print("❌ Error:", e)
    
    print("\nYOLOv3 implementation completed!")

fn run_component_tests() raises:
    """Run tests for individual components."""
    print("Running component tests...")
    
    # Step 1: Test basic tensor operations
    print("\nStep 1: Tensor Operations Test")
    print("-----------------------------")
    test_tensor_ops()
    
    # Step 2: Test matrix operations
    print("\nStep 2: Matrix Operations Test")
    print("-----------------------------")
    matrix_ops.test_matrix_ops()
    
    # Step 3: Test convolutional layers
    print("\nStep 3: Conv Layer Test")
    print("----------------------")
    conv_layers.test_conv_layers()
    
    # Step 4: Test utility functions
    print("\nStep 4: YOLO Utils Test")
    print("----------------------")
    yolo_utils.test_utils()

fn test_tensor_ops() raises:
    """Test basic tensor operations."""
    # Create a test tensor
    var spec = TensorSpec(DType.float32, 2, 3, 4)
    var tensor = Tensor[DType.float32](spec)
    
    # Initialize with values
    for i in range(tensor.num_elements()):
        tensor[i] = Float32(i)
    
    print("Created tensor with shape:", tensor.shape())
    print("First few elements:", tensor[0], tensor[1], tensor[2])