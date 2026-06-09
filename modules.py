import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlappingPatchEmbed(nn.Module):
    """
    Block 1: Overlapping Patch Tokenizer
    16x16 patches with stride 8 (50% overlap).
    """
    def __init__(self, in_chans=3, embed_dim=128, patch_size=16, stride=8):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)

    def forward(self, x):
        x = self.proj(x)
        return x  # Shape: (B, C, H, W)

class ReversePatchEmbed(nn.Module):
    """
    Block 7: Patch Reconstructor (Reverse overlap)
    """
    def __init__(self, embed_dim=128, out_chans=3, patch_size=16, stride=8):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=stride, padding=patch_size//2)

    def forward(self, x):
        return self.proj(x)

class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttentionGate(nn.Module):
    """
    Spatial Attention Module
    Produces an HxW map of values 0->1 representing complexity.
    """
    def __init__(self):
        super(SpatialAttentionGate, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        map_out = self.sigmoid(attention)
        return x * map_out, map_out

class CustomAttentionStack(nn.Module):
    """
    Combines a pseudo-Swin Window block with Spatial and Channel Attention.
    (Simplified for demonstration, in full code wrap timm's SwinBlock here).
    """
    def __init__(self, dim):
        super().__init__()
        # Standard local convolution as a proxy for Window Attention in this skeleton
        self.window_attn = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) 
        self.spatial_gate = SpatialAttentionGate()
        self.channel_attn = ChannelAttention(dim)
        
    def forward(self, x):
        x = self.window_attn(x)
        x, spatial_map = self.spatial_gate(x)
        x = self.channel_attn(x)
        return x, spatial_map
