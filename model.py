import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from modules import OverlappingPatchEmbed, ReversePatchEmbed, CustomAttentionStack


class AttentionGuidedSwinCompression(CompressionModel):
    """
    Attention-guided learned image compression model for 512x512 images.

    Shape path for 512x512 input:

    x:        B x 3   x 512 x 512
    patch:   B x N   x 64  x 64
    down:    B x M   x 32  x 32
    z:       B x N   x 8   x 8
    h_s:     B x ... x 32  x 32
    up:      B x N   x 64  x 64
    output:  B x 3   x 512 x 512
    """

    def __init__(self, N=128, M=192, gain_min=0.1):
        super().__init__()

        self.N = N
        self.M = M
        self.gain_min = gain_min

        # Analysis transform / encoder
        self.patch_embed = OverlappingPatchEmbed(
            in_chans=3,
            embed_dim=N,
            patch_size=16,
            stride=8
        )

        self.enc_stage1 = CustomAttentionStack(N)

        self.enc_down1 = nn.Conv2d(
            N,
            M,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.enc_stage2 = CustomAttentionStack(M)

        # Hyperprior
        self.entropy_bottleneck = EntropyBottleneck(N)

        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1),  # 16 -> 8
        )

        # Output:
        # M channels for scales
        # M channels for means
        # 1 channel for spatial gain map
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(
                N,
                N,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                N,
                (2 * M) + 1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),  # 16 -> 32
        )

        self.gaussian_conditional = GaussianConditional(None)

        # Synthesis transform / decoder
        self.dec_stage2 = CustomAttentionStack(M)

        self.dec_up1 = nn.ConvTranspose2d(
            M,
            N,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )  # 32 -> 64

        self.dec_stage1 = CustomAttentionStack(N)

        self.patch_reconstruct = ReversePatchEmbed(
            embed_dim=N,
            out_chans=3,
            patch_size=16,
            stride=8
        )

    def _crop_like(self, src, target):
        """
        Ensures src spatial size matches target spatial size.
        Useful for safety if dimensions change later.
        """
        return src[:, :, :target.size(2), :target.size(3)]

    def forward(self, x):
        # -------------------------
        # Encoder
        # -------------------------
        y = self.patch_embed(x)
        y, spatial_map_enc1 = self.enc_stage1(y)

        y = self.enc_down1(y)
        y, spatial_map_enc2 = self.enc_stage2(y)

        # y shape should be B x M x 32 x 32 for 512x512 input

        # -------------------------
        # Hyperprior
        # -------------------------
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        hyper_params = self.h_s(z_hat)
        hyper_params = self._crop_like(hyper_params, y)

        scales_hat = hyper_params[:, :self.M, :, :]
        means_hat = hyper_params[:, self.M:2 * self.M, :, :]
        gain_logits = hyper_params[:, 2 * self.M:2 * self.M + 1, :, :]

        # Decoder-available spatial gain map.
        # Range: [gain_min, 1 + gain_min]
        gain_map = torch.sigmoid(gain_logits) + self.gain_min

        # Attention-guided adaptive latent scaling
        y_scaled = y * gain_map

        # Entropy model for scaled latent
        y_hat, y_likelihoods = self.gaussian_conditional(
            y_scaled,
            scales_hat,
            means=means_hat
        )

        # Inverse scaling.
        # Valid because gain_map comes from z_hat, which decoder can also access.
        y_hat_unscaled = y_hat / gain_map

        # -------------------------
        # Decoder
        # -------------------------
        x_hat, _ = self.dec_stage2(y_hat_unscaled)
        x_hat = self.dec_up1(x_hat)
        x_hat, _ = self.dec_stage1(x_hat)

        x_hat = self.patch_reconstruct(x_hat)

        # Safety crop for exact 512x512 or any input size.
        x_hat = x_hat[:, :, :x.size(2), :x.size(3)]

        # Keep reconstruction in valid image range.
        x_hat = torch.sigmoid(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            },
            "spatial_map": gain_map,
            "encoder_attention_map": spatial_map_enc2
        }