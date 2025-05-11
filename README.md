# waymojo
Manoj, Akil + Sujitra's Modular attempt at yoloing mojo.

# YOLOv3 in Mojo

A high-performance implementation of YOLOv3 object detection in Mojo, optimized for MAX Engine.

## Latest Fixes (Trait Implementation Update)

We've implemented several critical fixes to make the codebase compatible with the latest Mojo version:

1. **Copyable & Movable Trait Implementation**:
   - Added proper `__copyinit__` and `__moveinit__` methods to Conv2DBlock and ResidualBlock
   - Fixed ownership semantics for collections of structs (List[Conv2DBlock], List[ResidualBlock])
   - Implemented explicit deep copy methods to handle complex struct relationships

2. **Optional Type Handling**:
   - Fixed Optional[Conv2DBlock] handling in ResidualBlock
   - Implemented correct ownership semantics for moving Optional values
   - Added proper initialization of Optional fields

3. **Time Module Migration**:
   - Updated from `now()` to `perf_counter()` for timing
   - Fixed time unit calculations (now in seconds instead of nanoseconds)
   - Updated benchmarking calculations to use proper time units

4. **Constructor Fixes**:
   - Updated all constructors to use `out self` instead of `inout self`
   - Ensured proper field initialization before use
   - Fixed struct initialization in collections

These changes ensure compatibility with the latest Mojo releases and enable proper use of YOLOv3 in your projects.

## Project Structure
