import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# SigLIP-style patch-embedding stem
# ---------------------------------------------------------------------------
# Replaces the old strided-CNN + global-average-pool encoder, which destroyed
# the spatial "where is the line" signal a line-follower depends on.
#
# A single strided convolution splits the image into a grid of non-overlapping
# patches and linearly embeds each into D_MODEL — exactly the ViT / SigLIP patch
# stem. The output KEEPS the spatial grid as a sequence of tokens so the
# downstream transformers can reason about line position.
#
#   image (B, 3, H, W)  --conv(patch,patch)-->  (B, D, H/patch, W/patch)
#                       --flatten-->             (B, N_patches, D)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Conv patch-embed stem: image -> sequence of patch tokens (no pooling)."""

    def __init__(self, in_channels: int = 3, patch_size: int = 8, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        # A stride==kernel conv is a linear projection of each non-overlapping patch.
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)
        returns : (B, N_patches, d_model) where N_patches = (H/patch)*(W/patch)
        """
        if x.ndim != 4:
            raise ValueError(f"PatchEmbed expects a 4D tensor (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.proj.in_channels:
            raise ValueError(
                f"PatchEmbed expects {self.proj.in_channels} image channels, got {x.shape[1]}"
            )
        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError(
                f"Image height/width must be divisible by patch_size={self.patch_size}, "
                f"got {(x.shape[2], x.shape[3])}"
            )
        x = self.proj(x)                    # (B, D, Hp, Wp)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, Hp*Wp, D)
        return self.norm(x)


# Backwards-compatible alias — some call sites imported ConvolutionalEncoder.
# It now returns patch tokens instead of a pooled feature map.
ConvolutionalEncoder = PatchEmbed


def image_to_tensor(image_np):
    """HWC uint8 image (or PIL) -> (1, 3, H, W) float tensor in [0, 1]."""
    if image_np is None:
        return None

    if isinstance(image_np, Image.Image):
        import numpy as np
        image_np = np.array(image_np)

    if image_np.ndim == 2:
        image_np = image_np[:, :, None].repeat(3, axis=2)
    if image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]
    if image_np.shape[2] != 3:
        raise ValueError(f"image_to_tensor expects 1, 3, or 4 channels, got {image_np.shape[2]}")

    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0)  # add batch dimension


if __name__ == '__main__':
    stem = PatchEmbed(in_channels=3, patch_size=8, d_model=256)
    dummy = torch.randn(1, 3, 64, 64)
    out = stem(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Tokens : {tuple(out.shape)}")   # (1, 64, 256)  = 8x8 grid, d=256
