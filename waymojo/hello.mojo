# ===== hello.mojo =====
# Simple test to verify basic Mojo functionality
from builtin.math import min, max
from time import perf_counter_ns

fn main():
    print("Hello, Mojo!")
    
    var a = 5
    var b = 7
    print("a + b =", a + b)
    
    # Test min/max functions
    var min_val = min(10, 20)
    var max_val = max(10, 20)
    print("min(10, 20) =", min_val)
    print("max(10, 20) =", max_val)
    
    # Test float min/max
    fn max_float(a: Float32, b: Float32) -> Float32:
        return a if a > b else b
    
    var max_float_val = max_float(10.5, 20.5)
    print("max_float(10.5, 20.5) =", max_float_val)
    
    # Test timing function
    var start_time = perf_counter_ns()
    # Do some work
    for i in range(1000000):
        var x = i * i
    var end_time = perf_counter_ns()
    
    print("Time taken:", (end_time - start_time) / 1e6, "ms")
    
    print("Basic Mojo test completed!") 