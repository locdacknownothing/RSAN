import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Input channel is 2 because we concatenate average and max pooled features
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # He normal initialization is typical for Keras, we replicate it here
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='sigmoid')

    def forward(self, x):
        # Input shape: [B, C, H, W]
        # Pool along the channel dimension (dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate average and max pool outputs along channel axis
        concat = torch.cat([avg_out, max_out], dim=1)
        # Pass through Conv2d and Sigmoid activation
        attention = self.conv(concat)
        attention = torch.sigmoid(attention)
        # Element-wise multiply input features with attention mask
        return x * attention
