
import gpu
import layout

def convolution_kernel[T: DType,
                       W: Int32, # input/output
                       H: Int32,
                       C: Int32,
                       w: Int32, # kernel
                       h: Int32,
                       c: Int32]
(
    kernel: layout.LayoutTensor[T, layout.Layout.row_major(c,h,w), MutableAnyOrigin],
    input: layout.LayoutTensor[T, layout.Layout.row_major(C, H, W), MutableAnyOrigin],
    output: layout.LayoutTensor[T, layout.Layout.row_major(c, H, W), MutableAnyOrigin] # must be initialized to 0
) -> None:
    
    alias mid_w = w // 2
    alias mid_h = h // 2
    y = gpu.id.global.y
    x = gpu.id.global.x
    for i in range(-mid_h, mid_h + 1):
        for j in range(-mid_w, mid_w + 1):
            if (y + i >= 0 and y + i < H) and (x + j >= 0 and x + j < W):
                for CC in range(C):
                    for cc in range(c):
                        output[cc,i,j] += input[CC, y + i, x + j] * kernel[cc, i + mid_h, j + mid_w]
                    #:
                #:
            #:
        #:for CC
    #:for cc
#:convolution_kernel()


def enqueue_convolution[T: DType,
                        W: Int32, # input/output
                        H: Int32,
                        C: Int32,
                        w: Int32, # kernel
                        h: Int32,
                        c: Int32]
(
    ctx: gpu.host.DeviceContext,
    input: gpu.host.HostBuffer[DType.Float32],
    output: gpu.host.HostBuffer[DType.Float32],
    conv_kernel: List[DType.Float32]
) -> None:

    ck_device = ctx.enqueue_create_buffer[T](w*h)
    ctx.synchronize()
    with ck_device.map_to_host() as ck_host:
        for i in range(N):
            ck_host[i] = conv_kernel[i]
        #:
    #:
    ck_device_layout = layout.LayoutTensor[T, conv_kernel_layout](ck_device)
    io_layout = layout.Layout.row_major(h, w, c)
    input_device = ctx.enqueue_create_buffer[T](len(input))
    input_device_layout = layout.LayoutTensor[T, io_layout](input_device)
    output_device_layout = layout.LayoutTensor[T, io_layout](output_device)
    output_device = ctx.enqueue_create_buffer[DType.Float32](len(output))
    ctx.enqueue_copy(input, input_device)
    alias BLOCK_DIM_X = 16
    alias BLOCK_DIM_Y = 16
    ctx.enqueue_function[convolution_kernel](ck_device, input_device, output_device,
                                             grid_dim=(ceildiv(h, BLOCK_DIM_X), ceildiv(w, BLOCK_DIM_Y)),
                                             block_dim=(BLOCK_DIM_X, BLOCK_DIM_Y))
    ctx.enqueue_copy(output_device, output)
    ctx.enqueue_copy()

#:enqueue_convolution
