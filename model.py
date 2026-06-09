import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from modules import OverlappingPatchEmbed, ReversePatchEmbed, CustomAttentionStack

class AttentionGuidedSwinCompression(CompressionModel):
    def __init__(self, N=128, M=192):
        super().__init__()
        # N: channel dimension of intermediate latents
        # M: channel dimension of bottleneck latent
        
        # Block 1
        self.patch_embed = OverlappingPatchEmbed(in_chans=3, embed_dim=N)
        
        # Block 2 (Encoder) - Simplified stages
        self.enc_stage1 = CustomAttentionStack(N)
        self.enc_down1 = nn.Conv2d(N, M, stride=2, kernel_size=3, padding=1)
        self.enc_stage2 = CustomAttentionStack(M)
        
        # Block 4 - Hyperprior Network
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1),
        )
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N, M * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        
        # Block 6 (Decoder)
        self.dec_stage2 = CustomAttentionStack(M)
        self.dec_up1 = nn.ConvTranspose2d(M, N, stride=2, kernel_size=3, padding=1, output_padding=1)
        self.dec_stage1 = CustomAttentionStack(N)
        
        # Block 7
        self.patch_reconstruct = ReversePatchEmbed(embed_dim=N, out_chans=3)

    def forward(self, x):
        # Encode
        y = self.patch_embed(x)
        y, _ = self.enc_stage1(y)
        y = self.enc_down1(y)
        y, spatial_map = self.enc_stage2(y)  # Extracted attention map
        
        # Block 3: Adaptive Quantizer (Mock representation)
        # We scale the latent by the spatial attention map before entropy coding
        # This forces the network to allocate bits based on spatial importance
        y_scaled = y * (spatial_map + 0.1) # Add small epsilon to avoid zeroing out completely

        # Hyperprior
        z = self.h_a(torch.abs(y_scaled))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y_scaled, scales_hat, means=means_hat)
        
        # Inverse scaling (Decoder side)
        # Note: in a real codec, the spatial map must be transmitted or derived from hyperprior.
        # Here we approximate its recovery from the decoded latent.
        y_hat_unscaled = y_hat / (spatial_map + 0.1)

        # Decode
        x_hat, _ = self.dec_stage2(y_hat_unscaled)
        x_hat = self.dec_up1(x_hat)
        x_hat, _ = self.dec_stage1(x_hat)
        x_hat = self.patch_reconstruct(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "spatial_map": spatial_map
        }
