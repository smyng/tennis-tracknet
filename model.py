import torch
import torch.nn as nn

class Conv2DBlock(nn.Module):
    """ Conv2D + BN + ReLU """
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNet, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
        x = self.bottleneck(x)                                          # (N,  512,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
        x = self.up_block_1(x)                                          # (N,  256,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
        x = self.up_block_2(x)                                          # (N,  128,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
        x = self.up_block_3(x)                                          # (N,   64,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x


class TrackNet2x(nn.Module):
    """TrackNet variant for 2x input resolution (576x1024).

    Adds one extra encoder/decoder stage at the full resolution (576x1024)
    with lightweight 48-channel convolutions, then pools down to 288x512
    where the standard 3-stage encoder operates identically to the original.
    The bottleneck stays at 36x64, keeping total compute ~1.6x of original.

    Architecture:
        Encoder:
            stage0: Double2DConv(in_dim->48) @ 576x1024 + MaxPool
            stage1: Double2DConv(48->64)     @ 288x512  + MaxPool
            stage2: Double2DConv(64->128)    @ 144x256  + MaxPool
            stage3: Triple2DConv(128->256)   @ 72x128   + MaxPool
        Bottleneck: Triple2DConv(256->512)   @ 36x64
        Decoder (mirror + skip connections):
            up1: Triple2DConv(768->256)      @ 72x128
            up2: Double2DConv(384->128)      @ 144x256
            up3: Double2DConv(192->64)       @ 288x512
            up4: Double2DConv(112->48)       @ 576x1024
        Output: Conv2d(48->out_dim) + Sigmoid
    """
    def __init__(self, in_dim, out_dim):
        super(TrackNet2x, self).__init__()
        # Stage 0: high-res front-end (576x1024)
        self.down_block_0 = Double2DConv(in_dim, 48)
        # Stages 1-3: same structure as original TrackNet
        self.down_block_1 = Double2DConv(48, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        # Decoder
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.up_block_4 = Double2DConv(64 + 48, 48)  # concat with stage0 skip
        self.predictor = nn.Conv2d(48, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.down_block_0(x)                                       # (N,   48,  576,  1024)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x0)                     # (N,   48,  288,   512)
        x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
        x = self.bottleneck(x)                                          # (N,  512,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
        x = self.up_block_1(x)                                          # (N,  256,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
        x = self.up_block_2(x)                                          # (N,  128,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
        x = self.up_block_3(x)                                          # (N,   64,  288,   512)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x0], dim=1)      # (N,  112,  576,  1024)
        x = self.up_block_4(x)                                          # (N,   48,  576,  1024)
        x = self.predictor(x)                                           # (N, out_dim, 576, 1024)
        x = self.sigmoid(x)
        return x


class MotionAttention(nn.Module):
    """
    Learnable motion attention from TrackNetV4.

    Computes frame-to-frame differences, then applies a learnable power
    transform to produce an attention map that highlights moving regions.
    The attention is applied via element-wise multiplication at the
    bottleneck, focusing the model on motion (where the ball likely is).

    Adds only 2 learnable parameters (gain + bias for the power transform).
    """
    def __init__(self):
        super(MotionAttention, self).__init__()
        # Learnable power transform: sigmoid(gain * motion + bias)
        # Initialized so the transform is close to identity at start
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.pool = nn.AdaptiveAvgPool2d(None)  # placeholder, set dynamically

    def forward(self, input_frames, bottleneck_h, bottleneck_w, channels_per_frame):
        """
        Args:
            input_frames: Raw model input (N, in_dim, H, W)
            bottleneck_h: Height of bottleneck feature map
            bottleneck_w: Width of bottleneck feature map
            channels_per_frame: Channels per frame (3 for RGB, 4 for subtract_concat)

        Returns:
            attention: (N, 1, bottleneck_h, bottleneck_w) attention map
        """
        N, C, H, W = input_frames.shape
        num_frames = C // channels_per_frame

        # Extract grayscale for each frame: average over RGB channels (first 3 of each group)
        grays = []
        for f in range(num_frames):
            rgb = input_frames[:, f*channels_per_frame : f*channels_per_frame+3, :, :]  # (N, 3, H, W)
            gray = rgb.mean(dim=1, keepdim=True)  # (N, 1, H, W)
            grays.append(gray)

        # Compute frame-to-frame differences and take max across all pairs
        if len(grays) < 2:
            # Single frame — no motion, return ones (no-op attention)
            return torch.ones(N, 1, bottleneck_h, bottleneck_w, device=input_frames.device)

        diffs = []
        for i in range(len(grays) - 1):
            diff = (grays[i+1] - grays[i]).abs()  # (N, 1, H, W)
            diffs.append(diff)

        # Aggregate: max across all frame pairs to capture any motion
        motion = torch.stack(diffs, dim=0).max(dim=0)[0]  # (N, 1, H, W)

        # Downsample to bottleneck resolution
        motion = nn.functional.adaptive_avg_pool2d(motion, (bottleneck_h, bottleneck_w))

        # Learnable power transform
        attention = torch.sigmoid(self.gain * motion + self.bias)

        return attention


class TrackNetV4(nn.Module):
    """
    TrackNet with motion attention maps (V4 variant).

    Same U-Net architecture as TrackNet, plus a lightweight motion attention
    module that modulates bottleneck features based on frame-to-frame motion.
    This helps the model focus on moving objects (the ball) while suppressing
    static background.

    Can be initialized from a pretrained TrackNet checkpoint for fine-tuning.
    """
    def __init__(self, in_dim, out_dim):
        super(TrackNetV4, self).__init__()
        # Store for computing channels_per_frame in forward
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Same encoder-decoder as TrackNet
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

        # V4 addition: motion attention
        self.motion_attention = MotionAttention()

    def forward(self, x):
        raw_input = x  # save for motion computation

        x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
        x = self.bottleneck(x)                                          # (N,  512,   36,    64)

        # V4: Apply motion attention at bottleneck
        channels_per_frame = self.in_dim // self.out_dim
        attn = self.motion_attention(
            raw_input, x.shape[2], x.shape[3], channels_per_frame
        )                                                                # (N,    1,   36,    64)
        x = x * attn                                                    # (N,  512,   36,    64)

        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
        x = self.up_block_1(x)                                          # (N,  256,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
        x = self.up_block_2(x)                                          # (N,  128,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
        x = self.up_block_3(x)                                          # (N,   64,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x

    def load_tracknet_weights(self, tracknet_state_dict):
        """Load weights from a pretrained TrackNet (V3) checkpoint for fine-tuning."""
        own_state = self.state_dict()
        loaded = 0
        for name, param in tracknet_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"Loaded {loaded}/{len(tracknet_state_dict)} weights from TrackNet checkpoint "
              f"(skipped {len(tracknet_state_dict) - loaded}, V4 motion attention initialized fresh)")

    
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Double1DConv(nn.Module):
    """ Conv1DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double1DConv, self).__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)                                   # (N,   L,   3)
        x = x.permute(0, 2, 1)                                         # (N,   3,   L)
        x1 = self.down_1(x)                                            # (N,  16,   L)
        x2 = self.down_2(x1)                                           # (N,  32,   L)
        x3 = self.down_3(x2)                                           # (N,  64,   L)
        x = self.buttleneck(x3)                                        # (N,  256,  L)
        x = torch.cat([x, x3], dim=1)                                  # (N,  384,  L)
        x = self.up_1(x)                                               # (N,  128,  L)
        x = torch.cat([x, x2], dim=1)                                  # (N,  192,  L)
        x = self.up_2(x)                                               # (N,   64,  L)
        x = torch.cat([x, x1], dim=1)                                  # (N,   96,  L)
        x = self.up_3(x)                                               # (N,   32,  L)
        x = self.predictor(x)                                          # (N,   2,   L)
        x = self.sigmoid(x)                                            # (N,   2,   L)
        x = x.permute(0, 2, 1)                                         # (N,   L,   2)
        return x