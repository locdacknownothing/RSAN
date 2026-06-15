import torch
import torch.nn as nn
from Dropblock import DropBlock2D
from attention_module import SpatialAttention

class BatchActivate(nn.Module):
    def __init__(self, num_features):
        super(BatchActivate, self).__init__()
        # eps=2e-5, momentum=0.1 (matches momentum=0.9 in Keras)
        self.bn = nn.BatchNorm2d(num_features, eps=2e-05, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(x))

class ConvDropBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation=True, keep_prob=0.9, block_size=7):
        super(ConvDropBlock, self).__init__()
        
        # In PyTorch, padding='same' is supported from version 1.9 for stride=1
        # For maximum compatibility with older/other versions, we calculate the padding manually
        if padding == 'same':
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
            else:
                pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            pad = 0
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=True)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.activation = activation
        
        if self.activation:
            self.batch_activate = BatchActivate(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropblock(x)
        if self.activation:
            x = self.batch_activate(x)
        return x

class ResidualDropBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_activate=False, keep_prob=0.9, block_size=7):
        super(ResidualDropBlock, self).__init__()
        self.batch_activate_input = BatchActivate(in_channels)
        self.conv1 = ConvDropBlock(in_channels, out_channels, (3, 3), keep_prob=keep_prob, block_size=block_size)
        self.conv2 = ConvDropBlock(out_channels, out_channels, (3, 3), activation=False, keep_prob=keep_prob, block_size=block_size)
        
        # Linear projection shortcut if input channels != output channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), padding=0, bias=True)
        else:
            self.shortcut = None
            
        self.batch_activate = batch_activate
        if self.batch_activate:
            self.batch_activate_output = BatchActivate(out_channels)

    def forward(self, blockInput):
        x = self.batch_activate_input(blockInput)
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.shortcut is not None:
            shortcut_val = self.shortcut(blockInput)
        else:
            shortcut_val = blockInput
            
        x = x + shortcut_val
        if self.batch_activate:
            x = self.batch_activate_output(x)
        return x

class RSAB(nn.Module):
    def __init__(self, channels, block_size=7, keep_prob=0.9):
        super(RSAB, self).__init__()
        self.batch_activate_input = BatchActivate(channels)
        self.conv1 = ConvDropBlock(channels, channels, (3, 3), keep_prob=keep_prob, block_size=block_size)
        self.conv2 = ConvDropBlock(channels, channels, (3, 3), activation=False, keep_prob=keep_prob, block_size=block_size)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.batch_activate_output = BatchActivate(channels)

    def forward(self, input_tensor):
        f = self.batch_activate_input(input_tensor)
        f = self.conv1(f)
        f = self.conv2(f)
        x = self.spatial_attention(f)
        result = input_tensor + x
        result = self.batch_activate_output(result)
        return result
