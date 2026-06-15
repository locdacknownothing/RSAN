import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    """
    DropBlock2D: A regularization method for convolutional networks.
    Reference: https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self, block_size, keep_prob):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        # x has shape: [B, C, H, W]
        if not self.training or self.keep_prob >= 1.0:
            return x

        B, C, H, W = x.shape
        gamma = self._get_gamma(H, W)

        # 1. Sample mask from Bernoulli distribution with parameter gamma
        mask = (torch.rand((B, C, H, W), device=x.device) < gamma).to(x.dtype)

        # 2. Compute valid seed region
        half_block_size = self.block_size // 2
        valid_mask = torch.zeros((H, W), dtype=x.dtype, device=x.device)
        
        h_start = min(half_block_size, H)
        h_end = max(H - half_block_size, 0)
        w_start = min(half_block_size, W)
        w_end = max(W - half_block_size, 0)
        
        valid_mask[h_start:h_end, w_start:w_end] = 1.0
        
        # Apply the valid region constraint to mask
        mask = mask * valid_mask.unsqueeze(0).unsqueeze(0)

        # 3. Apply max pool with kernel_size = block_size to expand drop area
        padding = self.block_size // 2
        block_mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=padding
        )

        # Crop if shape does not match exactly
        if block_mask.shape[2:] != (H, W):
            block_mask = block_mask[:, :, :H, :W]

        # 4. Invert mask (1 for keep, 0 for drop)
        block_mask = 1.0 - block_mask

        # 5. Normalize the features to keep expectation constant
        normalize = block_mask.numel() / (block_mask.sum() + 1e-8)

        return x * block_mask * normalize

    def _get_gamma(self, height, width):
        height, width = float(height), float(width)
        block_size = float(self.block_size)
        
        # Prevent division by zero
        denom = (height - block_size + 1.0) * (width - block_size + 1.0)
        if denom <= 0:
            return 0.0
            
        return ((1.0 - self.keep_prob) / (block_size ** 2)) * ((height * width) / denom)
