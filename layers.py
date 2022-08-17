import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

"""
Weight Scaled Convolution : Equalized learning rate applied on conv2d layer / Used as initial conv for Generator and Discriminator
conv2d: pixel * weight <==> WSconv2d: {pixel * (1/norm)} * weight
"""
class WSconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (kernel_size * kernel_size * in_channels)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # ininialize weights & bias
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        ## shape : [batch_size, C, H, W]
        return x / torch.square(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

"""
ConvBlock : Used by both Generator and Discriminator / 'pixel_norm' is True only in Generator
Order of operations : "conv2d -> activation -> norm"
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSconv2d(in_channels, out_channels)
        self.conv2 = WSconv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.use_pixelnorm = use_pixelnorm
        self.pixelnorm = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pixelnorm(x) if self.use_pixelnorm else x
        x = self.leaky(self.conv2(x))
        x = self.pixelnorm(x) if self.use_pixelnorm else x
        return x