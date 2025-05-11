# ===== math_utils.mojo =====
# Common math utility functions to replace missing functions in newer Mojo versions
from builtin.math import max, min
from time import perf_counter_ns

# Simple exponential function implementation
fn exp(x: Float32) -> Float32:
    """Exponential function (e^x)."""
    # Taylor series approximation for e^x
    var result: Float32 = 1.0
    var term: Float32 = 1.0
    var n: Int = 1
    
    # Use first 10 terms of the series for reasonable accuracy
    for i in range(1, 10):
        term *= x / i
        result += term
    
    return result

# Use perf_counter_ns as a replacement for the old now() function
fn now() -> Int:
    """Return current time in nanoseconds."""
    return perf_counter_ns() 