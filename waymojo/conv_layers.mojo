# ===== conv_layers.mojo =====
from tensor import Tensor, TensorSpec, TensorShape
from algorithm import parallelize
from math import sqrt
from builtin.math import max, min
from time import perf_counter_ns
import math_utils  # Use our custom math utilities

alias DT = DType.float32

fn leaky_relu(x: Tensor[DT], alpha: Float32 = 0.1) -> Tensor[DT]:
    """Apply LeakyReLU activation function."""
    var output = Tensor[DT](x.shape())
    
    # Sequential implementation instead of parallelized
    for i in range(x.num_elements()):
        var val = x[i]
        output[i] = val if val > 0 else alpha * val
    
    return output

fn batch_norm_2d(
    input: Tensor[DT],
    weight: Tensor[DT],
    bias: Tensor[DT],
    running_mean: Tensor[DT],
    running_var: Tensor[DT],
    eps: Float32 = 1e-5,
    training: Bool = True
) -> Tensor[DT]:
    """Batch normalization for 2D inputs."""
    # Input: [batch, channels, height, width]
    var batch = input.shape()[0]
    var channels = input.shape()[1]
    var height = input.shape()[2]
    var width = input.shape()[3]
    
    var output = Tensor[DT](input.shape())
    
    for c in range(channels):
        var mean: Float32 = 0.0
        var var_val: Float32 = 0.0
        
        if training:
            # Compute batch statistics
            var count = batch * height * width
            
            # Compute mean
            for b in range(batch):
                for h in range(height):
                    for w in range(width):
                        mean += input[b * channels * height * width + c * height * width + h * width + w]
            mean /= count
            
            # Compute variance
            for b in range(batch):
                for h in range(height):
                    for w in range(width):
                        var diff = input[b * channels * height * width + c * height * width + h * width + w] - mean
                        var_val += diff * diff
            var_val /= count
        else:
            # Use running statistics
            mean = running_mean[c]
            var_val = running_var[c]
        
        # Apply normalization
        var std = sqrt(var_val + eps)
        
        # Sequential normalization instead of parallelized
        for idx in range(batch * channels * height * width):
            if (idx % (channels * height * width)) // (height * width) == c:
                var normalized = (input[idx] - mean) / std
                output[idx] = weight[c] * normalized + bias[c]
    
    return output

struct Conv2DBlock:
    """Convolutional block with Conv2D + BatchNorm + LeakyReLU."""
    var conv_weight: Tensor[DT]
    var bn_weight: Tensor[DT]
    var bn_bias: Tensor[DT]
    var bn_running_mean: Tensor[DT]
    var bn_running_var: Tensor[DT]
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var padding: Int
    
    # Constructor with proper out self syntax
    fn __init__(
        out self, 
        in_channels: Int, 
        out_channels: Int, 
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0
    ):
        # Initialize tensors with proper shapes
        self.conv_weight = Tensor[DT](TensorShape(out_channels, in_channels, kernel_size, kernel_size))
        self.bn_weight = Tensor[DT](TensorShape(out_channels))
        self.bn_bias = Tensor[DT](TensorShape(out_channels))
        self.bn_running_mean = Tensor[DT](TensorShape(out_channels))
        self.bn_running_var = Tensor[DT](TensorShape(out_channels))
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize with He initialization
        var fan_in = kernel_size * kernel_size * in_channels
        var std = sqrt(2.0 / Float32(fan_in))
        
        # Initialize weights
        for i in range(self.conv_weight.num_elements()):
            self.conv_weight[i] = (Float32((i * 7) % 100) / 50.0 - 1.0) * std
        
        # Initialize BN parameters
        for i in range(out_channels):
            self.bn_weight[i] = 1.0
            self.bn_bias[i] = 0.0
            self.bn_running_mean[i] = 0.0
            self.bn_running_var[i] = 1.0
    
    # Method with proper self syntax (not mutating self)
    fn forward(self, input: Tensor[DT], training: Bool = True) -> Tensor[DT]:
        """Forward pass through the convolutional block."""
        # Convolution
        var x = matrix_ops.conv2d_optimized(input, self.conv_weight, None, self.stride, self.padding)
        
        # Batch normalization
        x = batch_norm_2d(x, self.bn_weight, self.bn_bias, 
                         self.bn_running_mean, self.bn_running_var, 1e-5, training)
        
        # LeakyReLU
        x = leaky_relu(x)
        
        return x

fn test_conv_layers() raises:
    """Test convolutional layers."""
    print("Testing Conv2D block...")
    
    # Create a test input
    var input = Tensor[DT](TensorShape(1, 3, 32, 32))
    
    # Initialize with random values
    for i in range(input.num_elements()):
        input[i] = Float32((i * 5) % 100) / 100.0
    
    # Create and test a Conv2D block
    var conv_block = Conv2DBlock(3, 16, 3, 1, 1)
    
    # Forward pass
    var output = conv_block.forward(input, training=True)
    
    print("Input shape:", input.shape())
    print("Output shape:", output.shape())
    print("First output value:", output[0])

# Need to import matrix_ops for conv2d_optimized
import matrix_ops