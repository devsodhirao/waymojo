# ===== yolov3_model.mojo =====
from tensor import Tensor, TensorShape
from algorithm import parallelize, vectorize
from math import exp, sqrt
from builtin.math import min, max
from collections import List, Optional
from time import perf_counter

alias DT = DType.float32

# Import our building blocks
import conv_layers
import matrix_ops
import yolo_utils

struct ResidualBlock(Copyable, Movable):
    """Residual block with two convolutional layers."""
    var conv1: conv_layers.Conv2DBlock
    var conv2: conv_layers.Conv2DBlock
    var downsample: Optional[conv_layers.Conv2DBlock]
    
    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        stride: Int = 1
    ):
        # First conv: 1x1
        self.conv1 = conv_layers.Conv2DBlock(in_channels, out_channels, 1, 1, 0)
        
        # Second conv: 3x3
        self.conv2 = conv_layers.Conv2DBlock(out_channels, out_channels, 3, stride, 1)
        
        # Downsample if needed (for skip connection)
        if stride != 1 or in_channels != out_channels:
            self.downsample = conv_layers.Conv2DBlock(in_channels, out_channels, 1, stride, 0)
        else:
            self.downsample = None
    
    # Copy constructor for Copyable trait
    fn __copyinit__(out self, existing: Self):
        # Deep copy the Conv2DBlock objects
        self.conv1 = existing.conv1.copy()
        self.conv2 = existing.conv2.copy()
        
        # Handle Optional field properly
        if existing.downsample:
            var ds = existing.downsample.value().copy()
            self.downsample = ds
        else:
            self.downsample = None
    
    # Move constructor for Movable trait
    fn __moveinit__(out self, owned existing: Self):
        # Move the Conv2DBlock objects
        self.conv1 = existing.conv1^
        self.conv2 = existing.conv2^
        
        # Use direct assignment for Optional field (will correctly handle the move)
        self.downsample = existing.downsample^
    
    fn forward(self, input: Tensor[DT], training: Bool = True) -> Tensor[DT]:
        """Forward pass with residual connection."""
        var identity = input
        
        # Main path
        var x = self.conv1.forward(input, training)
        x = self.conv2.forward(x, training)
        
        # Skip connection
        if self.downsample:
            identity = self.downsample.value().forward(input, training)
        
        # Add residual connection
        # Note: This is simplified - proper implementation would use element-wise addition
        for i in range(x.num_elements()):
            x[i] += identity[i]
        
        return conv_layers.leaky_relu(x, 0.1)

struct Darknet53:
    """Darknet-53 backbone for YOLOv3."""
    var conv1: conv_layers.Conv2DBlock
    var layer1: List[ResidualBlock]
    var layer2: List[ResidualBlock]
    var layer3: List[ResidualBlock]
    var layer4: List[ResidualBlock]
    var layer5: List[ResidualBlock]
    
    fn __init__(out self):
        # Initialize all fields
        self.conv1 = conv_layers.Conv2DBlock(3, 32, 3, 1, 1)
        
        # Initialize empty lists for layers
        self.layer1 = List[ResidualBlock]()
        self.layer2 = List[ResidualBlock]()
        self.layer3 = List[ResidualBlock]()
        self.layer4 = List[ResidualBlock]()
        self.layer5 = List[ResidualBlock]()
        
        # Create residual layers
        var layer1_blocks = self._make_layer(32, 64, 1, 2)   # 1 block, downsample
        var layer2_blocks = self._make_layer(64, 128, 2, 2)  # 2 blocks, downsample
        var layer3_blocks = self._make_layer(128, 256, 8, 2) # 8 blocks, downsample
        var layer4_blocks = self._make_layer(256, 512, 8, 2) # 8 blocks, downsample
        var layer5_blocks = self._make_layer(512, 1024, 4, 2) # 4 blocks, downsample
        
        # Assign blocks to the instance fields
        self.layer1 = layer1_blocks
        self.layer2 = layer2_blocks
        self.layer3 = layer3_blocks
        self.layer4 = layer4_blocks
        self.layer5 = layer5_blocks
    
    fn _make_layer(
        self,
        in_channels: Int,
        out_channels: Int,
        blocks: Int,
        stride: Int
    ) -> List[ResidualBlock]:
        """Create a layer with multiple residual blocks."""
        var layers = List[ResidualBlock]()
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return layers
    
    fn forward(self, input: Tensor[DT], training: Bool = True) -> List[Tensor[DT]]:
        """Forward pass returning features at multiple scales."""
        var features = List[Tensor[DT]]()
        
        # Initial convolution
        var x = self.conv1.forward(input, training)
        
        # Pass through residual layers
        x = self._forward_layer(self.layer1, x, training)
        x = self._forward_layer(self.layer2, x, training)
        
        # Store features for different scales
        x = self._forward_layer(self.layer3, x, training)
        features.append(x)  # 8x downsampling
        
        x = self._forward_layer(self.layer4, x, training)
        features.append(x)  # 16x downsampling
        
        x = self._forward_layer(self.layer5, x, training)
        features.append(x)  # 32x downsampling
        
        return features
    
    fn _forward_layer(self, layer: List[ResidualBlock], input: Tensor[DT], training: Bool) -> Tensor[DT]:
        """Forward pass through a layer of residual blocks."""
        var x = input
        # Iterate using indices to avoid pointer issues
        for i in range(len(layer)):
            var block = layer[i]
            x = block.forward(x, training)
        return x

struct YOLOv3Head:
    """YOLOv3 detection head."""
    var conv_sets: List[List[conv_layers.Conv2DBlock]]
    var final_convs: List[conv_layers.Conv2DBlock]
    var upsample_convs: List[conv_layers.Conv2DBlock]
    var num_classes: Int
    var num_anchors: Int
    
    fn __init__(out self, num_classes: Int):
        self.num_classes = num_classes
        self.num_anchors = 3  # 3 anchors per scale
        
        # Detection conv sets for each scale
        self.conv_sets = List[List[conv_layers.Conv2DBlock]]()
        self.final_convs = List[conv_layers.Conv2DBlock]()
        self.upsample_convs = List[conv_layers.Conv2DBlock]()
        
        # Scale 0 (32x): 1024 -> 512
        var conv_set_0 = List[conv_layers.Conv2DBlock]()
        conv_set_0.append(conv_layers.Conv2DBlock(1024, 512, 1))
        conv_set_0.append(conv_layers.Conv2DBlock(512, 1024, 3, 1, 1))
        conv_set_0.append(conv_layers.Conv2DBlock(1024, 512, 1))
        conv_set_0.append(conv_layers.Conv2DBlock(512, 1024, 3, 1, 1))
        conv_set_0.append(conv_layers.Conv2DBlock(1024, 512, 1))
        self.conv_sets.append(conv_set_0)
        
        # Final prediction layer for scale 0
        var output_filters = self.num_anchors * (5 + num_classes)
        self.final_convs.append(conv_layers.Conv2DBlock(512, 1024, 3, 1, 1))
        self.final_convs.append(conv_layers.Conv2DBlock(1024, output_filters, 1))
        
        # Scale 1 (16x): 768 -> 256
        var conv_set_1 = List[conv_layers.Conv2DBlock]()
        self.upsample_convs.append(conv_layers.Conv2DBlock(512, 256, 1))
        conv_set_1.append(conv_layers.Conv2DBlock(768, 256, 1))  # 512 + 256 (concatenated)
        conv_set_1.append(conv_layers.Conv2DBlock(256, 512, 3, 1, 1))
        conv_set_1.append(conv_layers.Conv2DBlock(512, 256, 1))
        conv_set_1.append(conv_layers.Conv2DBlock(256, 512, 3, 1, 1))
        conv_set_1.append(conv_layers.Conv2DBlock(512, 256, 1))
        self.conv_sets.append(conv_set_1)
        
        # Final prediction layer for scale 1
        self.final_convs.append(conv_layers.Conv2DBlock(256, 512, 3, 1, 1))
        self.final_convs.append(conv_layers.Conv2DBlock(512, output_filters, 1))
        
        # Scale 2 (8x): 384 -> 128
        var conv_set_2 = List[conv_layers.Conv2DBlock]()
        self.upsample_convs.append(conv_layers.Conv2DBlock(256, 128, 1))
        conv_set_2.append(conv_layers.Conv2DBlock(384, 128, 1))  # 256 + 128 (concatenated)
        conv_set_2.append(conv_layers.Conv2DBlock(128, 256, 3, 1, 1))
        conv_set_2.append(conv_layers.Conv2DBlock(256, 128, 1))
        conv_set_2.append(conv_layers.Conv2DBlock(128, 256, 3, 1, 1))
        conv_set_2.append(conv_layers.Conv2DBlock(256, 128, 1))
        self.conv_sets.append(conv_set_2)
        
        # Final prediction layer for scale 2
        self.final_convs.append(conv_layers.Conv2DBlock(128, 256, 3, 1, 1))
        self.final_convs.append(conv_layers.Conv2DBlock(256, output_filters, 1))
    
    fn forward(self, features: List[Tensor[DT]], training: Bool = True) -> List[Tensor[DT]]:
        """Forward pass through YOLOv3 head."""
        var predictions = List[Tensor[DT]]()
        
        # Process largest scale (32x)
        var x32 = features[2]
        # Use indexing to avoid pointer issues
        for i in range(len(self.conv_sets[0])):
            var conv = self.conv_sets[0][i]
            x32 = conv.forward(x32, training)
        
        var route_32 = x32
        var pred_32 = self.final_convs[0].forward(x32, training)
        pred_32 = self.final_convs[1].forward(pred_32, training)
        predictions.append(pred_32)
        
        # Process medium scale (16x)
        var x16_up = self.upsample_convs[0].forward(route_32, training)
        x16_up = self.upsample(x16_up)  # 2x upsampling
        var x16_concat = self.concatenate(x16_up, features[1])
        
        var x16 = x16_concat
        # Use indexing to avoid pointer issues
        for i in range(len(self.conv_sets[1])):
            var conv = self.conv_sets[1][i]
            x16 = conv.forward(x16, training)
        
        var route_16 = x16
        var pred_16 = self.final_convs[2].forward(x16, training)
        pred_16 = self.final_convs[3].forward(pred_16, training)
        predictions.append(pred_16)
        
        # Process smallest scale (8x)
        var x8_up = self.upsample_convs[1].forward(route_16, training)
        x8_up = self.upsample(x8_up)  # 2x upsampling
        var x8_concat = self.concatenate(x8_up, features[0])
        
        var x8 = x8_concat
        # Use indexing to avoid pointer issues
        for i in range(len(self.conv_sets[2])):
            var conv = self.conv_sets[2][i]
            x8 = conv.forward(x8, training)
        
        var pred_8 = self.final_convs[4].forward(x8, training)
        pred_8 = self.final_convs[5].forward(pred_8, training)
        predictions.append(pred_8)
        
        return predictions
    
    fn upsample(self, x: Tensor[DT]) -> Tensor[DT]:
        """Upsample feature map by 2x using nearest neighbor."""
        var batch = x.shape()[0]
        var channels = x.shape()[1]
        var height = x.shape()[2]
        var width = x.shape()[3]
        
        var output = Tensor[DT](TensorShape(batch, channels, height * 2, width * 2))
        
        @parameter
        fn upsample_element(idx: Int):
            var b = idx // (channels * height * 2 * width * 2)
            var remaining = idx % (channels * height * 2 * width * 2)
            var c = remaining // (height * 2 * width * 2)
            var remaining2 = remaining % (height * 2 * width * 2)
            var h = remaining2 // (width * 2)
            var w = remaining2 % (width * 2)
            
            var src_h = h // 2
            var src_w = w // 2
            var src_idx = b * channels * height * width + c * height * width + src_h * width + src_w
            
            output[idx] = x[src_idx]
        
        parallelize[upsample_element](output.num_elements())
        return output
    
    fn concatenate(self, tensor1: Tensor[DT], tensor2: Tensor[DT]) -> Tensor[DT]:
        """Concatenate two tensors along the channel dimension."""
        var batch = tensor1.shape()[0]
        var channels1 = tensor1.shape()[1]
        var channels2 = tensor2.shape()[1]
        var height = tensor1.shape()[2]
        var width = tensor1.shape()[3]
        
        var output = Tensor[DT](TensorShape(batch, channels1 + channels2, height, width))
        
        # Copy first tensor
        for b in range(batch):
            for c in range(channels1):
                for h in range(height):
                    for w in range(width):
                        var src_idx = b * channels1 * height * width + c * height * width + h * width + w
                        var dst_idx = b * (channels1 + channels2) * height * width + c * height * width + h * width + w
                        output[dst_idx] = tensor1[src_idx]
        
        # Copy second tensor
        for b in range(batch):
            for c in range(channels2):
                for h in range(height):
                    for w in range(width):
                        var src_idx = b * channels2 * height * width + c * height * width + h * width + w
                        var dst_idx = b * (channels1 + channels2) * height * width + (channels1 + c) * height * width + h * width + w
                        output[dst_idx] = tensor2[src_idx]
        
        return output

struct YOLOv3:
    """Complete YOLOv3 model."""
    var backbone: Darknet53
    var head: YOLOv3Head
    var num_classes: Int
    
    fn __init__(out self, num_classes: Int):
        self.num_classes = num_classes
        self.backbone = Darknet53()
        self.head = YOLOv3Head(num_classes)
    
    fn forward(self, input: Tensor[DT], training: Bool = True) -> List[Tensor[DT]]:
        """Forward pass through complete YOLOv3 model."""
        # Extract features
        var features = self.backbone.forward(input, training)
        
        # Generate predictions
        var predictions = self.head.forward(features, training)
        
        return predictions
    
    fn predict(self, image: Tensor[DT], conf_threshold: Float32 = 0.5) -> List[Tuple[Tensor[DT], Float32, Int]]:
        """Perform object detection on an image."""
        # Forward pass
        var predictions = self.forward(image, training=False)
        
        # Post-process predictions
        var detections = List[Tuple[Tensor[DT], Float32, Int]]()
        
        # Process each scale
        for scale_idx in range(3):
            var pred = predictions[scale_idx]
            var batch = pred.shape()[0]
            var grid_size = pred.shape()[2]
            var stride = 416 // grid_size  # Assuming 416x416 input
            
            # Get anchors for this scale
            var anchors = yolo_utils.get_anchors(scale_idx)
            
            # Process predictions for this scale
            var scale_detections = self._process_scale_predictions(
                pred, anchors, stride, grid_size, conf_threshold
            )
            
            detections.extend(scale_detections)
        
        # Apply NMS
        detections = self._apply_nms(detections, 0.5)
        
        return detections
    
    fn _process_scale_predictions(
        self,
        pred: Tensor[DT],
        anchors: List[Float32],
        stride: Int,
        grid_size: Int,
        conf_threshold: Float32
    ) -> List[Tuple[Tensor[DT], Float32, Int]]:
        """Process predictions for a single scale."""
        var detections = List[Tuple[Tensor[DT], Float32, Int]]()
        
        # Reshape predictions: [batch, (5+classes)*3, grid, grid] -> [batch, 3, 5+classes, grid, grid]
        # This is a simplified placeholder - actual reshaping would be more complex
        
        # Process each grid cell
        var batch = pred.shape()[0]
        var output_per_anchor = 5 + self.num_classes
        
        for b in range(batch):
            for i in range(grid_size):
                for j in range(grid_size):
                    for a in range(3):  # 3 anchors per cell
                        # Extract predictions (simplified)
                        var base_idx = b * (3 * output_per_anchor * grid_size * grid_size) + \
                                      a * (output_per_anchor * grid_size * grid_size) + \
                                      i * (output_per_anchor * grid_size) + \
                                      j * output_per_anchor
                        
                        # Get objectness score (apply sigmoid)
                        var obj_score = 1.0 / (1.0 + exp(-pred[base_idx + 4]))
                        
                        if obj_score >= conf_threshold:
                            # Extract box coordinates (simplified)
                            var box = Tensor[DT](TensorShape(4))
                            box[0] = pred[base_idx + 0]  # x
                            box[1] = pred[base_idx + 1]  # y
                            box[2] = pred[base_idx + 2]  # w
                            box[3] = pred[base_idx + 3]  # h
                            
                            # Find class with highest probability
                            var max_class_score: Float32 = 0.0
                            var max_class_id: Int = 0
                            
                            for c in range(self.num_classes):
                                var class_score = 1.0 / (1.0 + exp(-pred[base_idx + 5 + c]))
                                if class_score > max_class_score:
                                    max_class_score = class_score
                                    max_class_id = c
                            
                            var final_score = obj_score * max_class_score
                            
                            detections.append((box, final_score, max_class_id))
        
        return detections
    
    fn _apply_nms(
        self,
        detections: List[Tuple[Tensor[DT], Float32, Int]],
        iou_threshold: Float32
    ) -> List[Tuple[Tensor[DT], Float32, Int]]:
        """Apply non-maximum suppression."""
        # This is a simplified placeholder
        # Actual NMS would sort by confidence and remove overlapping boxes
        return detections