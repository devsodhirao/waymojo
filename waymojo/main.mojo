import sys
import gpu
from python import Python
import conv

def allocate_image_tensor[T: DType, C: Int32, H: Int32, W: Int32]
(
    ctx: gpu.host.DeviceContext,
    fill_with_zeros: Bool = False
) -> layout.LayoutTensor[T, layout.Layout.row_major(C, H, W)]:
    """
    Caller must ctx.synchronize() after this call, where necessary.
    """
    img_buffer = ctx.enqueue_create_buffer[T](C * H * W)
    img_layout = layout.Layout.row_major(C, H, W)
    img_tensor = layout.LayoutTensor[T, img_layout](img_buffer)
    if fill_with_zeros:
        ctx.enqueue_memset(img_buffer, T(0))
    #:
    return img_tensor
#:allocate_image_tensor()


def main():
	@parameter
	if not sys.has_accelerator():
		raise Error("No accelerator found")
	#:
    np = Python.import_module("numpy")
    pil = Python.import_module("PIL")
    pil_img = pil.Image.open("test.png")
    np_img = np.array(pil_img) 
    alias W = <known image W>
    alias H = <known image H>
	alias C = 3 # RGB
	alias DType = gpu.host.DType
	alias MutableAnyOrigin = gpu.host.MutableAnyOrigin

	# Convert to float32
	np_img = np_img.astype(np.float32)
	# Normalize to [0, 1]
	np_img /= 255.0
	img_tensor = layout.LayoutTensor[DType.uint8, layout.Layout.row_major(3, H, W)](np_img.unsafe_ptr())
	ctx = gpu.host.DeviceContext()	
	img_tensor_gpu = allocate_image_tensor[DType.uint8, 3, H, W](ctx)
    ctx.enqueue_copy(img_tensor, img_tensor_gpu)

	kernel = <import convolution kernel weights>
	conv.enqueue_convolution[DType.Float32, W, H, C, 3, 3, 32](ctx, img_tensor_gpu, output_tensor_gpu, kernel)

#:main()
