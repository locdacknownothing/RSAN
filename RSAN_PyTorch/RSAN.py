import torch
import torch.nn as nn
from layer import ResidualDropBlock, RSAB

class RSANetModel(nn.Module):
    def __init__(self, in_channels=3, start_neurons=16, keep_prob=0.9, block_size=7):
        super(RSANetModel, self).__init__()
        
        # Encoder
        self.conv1_1 = ResidualDropBlock(in_channels, start_neurons * 1, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.conv1_2 = RSAB(start_neurons * 1, keep_prob=keep_prob, block_size=block_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = ResidualDropBlock(start_neurons * 1, start_neurons * 2, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.conv2_2 = RSAB(start_neurons * 2, keep_prob=keep_prob, block_size=block_size)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = ResidualDropBlock(start_neurons * 2, start_neurons * 4, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.conv3_2 = RSAB(start_neurons * 4, keep_prob=keep_prob, block_size=block_size)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bridge (Middle)
        self.convm_1 = ResidualDropBlock(start_neurons * 4, start_neurons * 8, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.convm_2 = RSAB(start_neurons * 8, keep_prob=keep_prob, block_size=block_size)
        
        # Decoder
        # ConvTranspose2d stride=2 doubles size, kernel_size=3 & padding=1 & output_padding=1 gives exact 2x reconstruction
        self.deconv3 = nn.ConvTranspose2d(start_neurons * 8, start_neurons * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv3_1 = ResidualDropBlock(start_neurons * 8, start_neurons * 4, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.uconv3_2 = RSAB(start_neurons * 4, keep_prob=keep_prob, block_size=block_size)
        
        self.deconv2 = nn.ConvTranspose2d(start_neurons * 4, start_neurons * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv2_1 = ResidualDropBlock(start_neurons * 4, start_neurons * 2, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.uconv2_2 = RSAB(start_neurons * 2, keep_prob=keep_prob, block_size=block_size)
        
        self.deconv1 = nn.ConvTranspose2d(start_neurons * 2, start_neurons * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv1_1 = ResidualDropBlock(start_neurons * 2, start_neurons * 1, batch_activate=False, keep_prob=keep_prob, block_size=block_size)
        self.uconv1_2 = RSAB(start_neurons * 1, keep_prob=keep_prob, block_size=block_size)
        
        # Output layers
        self.output_layer = nn.Conv2d(start_neurons * 1, 1, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool3(conv3)
        
        # Bridge
        convm = self.convm_1(pool3)
        convm = self.convm_2(convm)
        
        # Decoder
        deconv3 = self.deconv3(convm)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.uconv3_1(uconv3)
        uconv3 = self.uconv3_2(uconv3)
        
        deconv2 = self.deconv2(uconv3)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.uconv2_1(uconv2)
        uconv2 = self.uconv2_2(uconv2)
        
        deconv1 = self.deconv1(uconv2)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.uconv1_1(uconv1)
        uconv1 = self.uconv1_2(uconv1)
        
        output = self.output_layer(uconv1)
        output = torch.sigmoid(output)
        
        return output

def RSANet(input_size=None, start_neurons=16, keep_prob=0.9, block_size=7, lr=1e-3):
    """
    Helper function to initialize the RSANet model, matching the API of the Keras implementation.
    """
    in_channels = 3 if input_size is None else input_size[2]
    model = RSANetModel(in_channels=in_channels, start_neurons=start_neurons, keep_prob=keep_prob, block_size=block_size)
    return model
