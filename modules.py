import torch
import torch.nn as nn


class OverlappingPatchEmbed(nn.Module):
    """
    Overlapping Patch Tokenizer.

    For 512x512 input with:
    patch_size=16, stride=8, padding=4

    Output size:
    512 -> 64

    Formula:
    out = floor((H + 2P - K) / S) + 1
        = floor((512 + 8 - 16) / 8) + 1
        = 64
    """
    def __init__(self, in_chans=3, embed_dim=128, patch_size=16, stride=8):
        super().__init__()
        padding = (patch_size - stride) // 2  # 4 for patch=16, stride=8

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.proj(x)


class ReversePatchEmbed(nn.Module):
    """
    Reverse overlapping patch reconstruction.

    For latent feature size 64x64:
    ConvTranspose2d with kernel=16, stride=8, padding=4

    Output:
    (64 - 1) * 8 - 2*4 + 16 = 512
    """
    def __init__(self, embed_dim=128, out_chans=3, patch_size=16, stride=8):
        super().__init__()
        padding = (patch_size - stride) // 2  # 4

        self.proj = nn.ConvTranspose2d(
            embed_dim,
            out_chans,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.proj(x)


class ChannelAttention(nn.Module):
    """
    CBAM-style channel attention.
    """
    def __init__(self, in_planes, ratio=16):
        super().__init__()

        hidden = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttentionGate(nn.Module):
    """
    Spatial attention gate producing a 1-channel spatial importance map.
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_out, max_out], dim=1)
        spatial_map = self.sigmoid(self.conv(attn))

        return x * spatial_map, spatial_map


class CustomAttentionStack(nn.Module):
    """
    Lightweight local-attention-inspired block.

    This is not a real Swin Transformer block yet.
    It uses:
    - depthwise convolution as local spatial mixing
    - spatial attention
    - channel attention
    - residual connection
    """
    def __init__(self, dim):
        super().__init__()

        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_gate = SpatialAttentionGate()
        self.channel_attn = ChannelAttention(dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x

        out = self.local_mixer(x)
        out = self.pointwise(out)
        out = self.norm(out)

        out, spatial_map = self.spatial_gate(out)
        out = self.channel_attn(out)

        out = out + residual

        return out, spatial_map