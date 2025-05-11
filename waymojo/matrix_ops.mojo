# ===== matrix_ops.mojo =====
from tensor import Tensor, TensorSpec, TensorShape
from algorithm import parallelize, vectorize
from builtin.math import min, max
from time import perf_counter_ns

alias DT = DType.float32
alias SIMD_WIDTH = 8  # Fixed value instead of simdwidthof

# Optimized matrix operations using proper Mojo Tensor API
fn matmul_optimized(A: Tensor[DT], B: Tensor[DT]) -> Tensor[DT]:
    """Optimized matrix multiplication using tiling and vectorization."""
    var M = A.shape()[0]
    var K = A.shape()[1]
    var N = B.shape()[1]
    
    var C = Tensor[DT](TensorShape(M, N))
    
    # Tile sizes for cache efficiency
    alias TILE_M = 64
    alias TILE_N = 64
    alias TILE_K = 64
    
    # Sequential implementation for now (to avoid parallelization issues)
    for i in range((M + TILE_M - 1) // TILE_M * ((N + TILE_N - 1) // TILE_N)):
        var tile_m = i // ((N + TILE_N - 1) // TILE_N)
        var tile_n = i % ((N + TILE_N - 1) // TILE_N)
        
        var m_start = tile_m * TILE_M
        var n_start = tile_n * TILE_N
        var m_end = min(m_start + TILE_M, M)
        var n_end = min(n_start + TILE_N, N)
        
        for m in range(m_start, m_end):
            for n in range(n_start, n_end):
                var sum: Float32 = 0.0  # Explicitly set type to Float32
                for k in range(K):
                    sum += A[m * K + k] * B[k * N + n]
                C[m * N + n] = sum
    
    return C

fn conv2d_optimized(
    input: Tensor[DT],
    weight: Tensor[DT],
    bias: Optional[Tensor[DT]] = None,
    stride: Int = 1,
    padding: Int = 0
) -> Tensor[DT]:
    """Optimized 2D convolution operation."""
    # Input: [batch, in_channels, height, width]
    # Weight: [out_channels, in_channels, kernel_height, kernel_width]
    
    var batch = input.shape()[0]
    var in_channels = input.shape()[1]
    var in_height = input.shape()[2]
    var in_width = input.shape()[3]
    
    var out_channels = weight.shape()[0]
    var kernel_height = weight.shape()[2]
    var kernel_width = weight.shape()[3]
    
    var out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    var out_width = (in_width + 2 * padding - kernel_width) // stride + 1
    
    var output = Tensor[DT](TensorShape(batch, out_channels, out_height, out_width))
    
    # Sequential implementation instead of parallelized
    for idx in range(batch * out_channels * out_height * out_width):
        var b = idx // (out_channels * out_height * out_width)
        var remaining = idx % (out_channels * out_height * out_width)
        var oc = remaining // (out_height * out_width)
        var remaining2 = remaining % (out_height * out_width)
        var oh = remaining2 // out_width
        var ow = remaining2 % out_width
        
        var sum = Float32(0.0)
        
        # Convolve
        for ic in range(in_channels):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    var ih = oh * stride - padding + kh
                    var iw = ow * stride - padding + kw
                    
                    if 0 <= ih < in_height and 0 <= iw < in_width:
                        var input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw
                        var weight_idx = oc * in_channels * kernel_height * kernel_width + ic * kernel_height * kernel_width + kh * kernel_width + kw
                        sum += input[input_idx] * weight[weight_idx]
        
        # Add bias if present
        if bias:
            sum += bias.value()[oc]
        
        var output_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow
        output[output_idx] = sum
    
    return output

fn test_matrix_ops() raises:
    """Test matrix operations."""
    print("Testing optimized matrix multiplication...")
    
    var M = 128
    var K = 128
    var N = 128
    
    var A = Tensor[DT](TensorShape(M, K))
    var B = Tensor[DT](TensorShape(K, N))
    
    # Initialize with random values
    for i in range(A.num_elements()):
        A[i] = Float32((i * 7) % 100) / 100.0
    
    for i in range(B.num_elements()):
        B[i] = Float32((i * 13) % 100) / 100.0
    
    # Time the operation
    var start_time = perf_counter_ns()
    var C = matmul_optimized(A, B)
    var end_time = perf_counter_ns()
    
    print("Matrix multiplication completed in:", (end_time - start_time) / 1e6, "ms")
    print("Output shape:", C.shape())
    print("First element:", C[0])