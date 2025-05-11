# ===== math_utils.mojo =====
# Common math utility functions to replace missing functions in newer Mojo versions
from builtin.math import max, min
from time import perf_counter
from tensor import Tensor, TensorShape

alias DT = DType.float32

# Simple exponential function implementation
fn exp(x: Float32) -> Float32:
    """Exponential function (e^x)."""
    # Taylor series approximation for e^x
    var result: Float32 = 1.0
    var term: Float32 = 1.0
    
    # Use first 10 terms of the series for reasonable accuracy
    for i in range(1, 10):
        term *= x / Float32(i)
        result += term
    
    return result

# Sigmoid function implementation 
fn sigmoid(x: Float32) -> Float32:
    """Sigmoid function: 1/(1+e^(-x))."""
    return 1.0 / (1.0 + exp(-x))

# Use perf_counter as a replacement for the old now() function
fn now() -> Float64:
    """Return current time in seconds."""
    return perf_counter() 