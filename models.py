import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import WSconv2d, ConvBlock, PixelNorm

# In generator conv blocks, the channels go like "512->512->512->512->256->128->64->32->16"
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        # starting structure of Generator should be opposite of ending structure of Discriminator
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSconv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb = WSconv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.progressive_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList([self.initial_rgb])  ## to rgb

        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            self.progressive_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, True))
            self.rgb_layers.append(WSconv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh((1 - alpha) * upscaled + alpha * generated)

    def forward(self, x, alpha, steps):  ## steps=0 : 4x4 output / steps=1 : 8x8 output / steps=2 : 16x16 output ...
        h = self.initial(x)

        if steps == 0:
            return self.initial_rgb(h)

        for step in range(steps):
            upscaled = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.progressive_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_generated = self.rgb_layers[steps](h)

        return self.fade_in(alpha, final_upscaled, final_generated)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()
        self.progressive_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList()  ## from rgb
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i - 1])

            ## rgb_layers list : [(1024x1024 input'from_rgb' layer),(512x512 input'from_rgb' layer),(256x256 input'from_rgb' layer) ...]
            self.rgb_layers.append(WSconv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))
            self.progressive_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixelnorm=False))

        ## this 'from-rgb' layer is for 4x4 resolution
        self.initial_rgb = WSconv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        ## ConvBlock for 4x4 resolution
        self.last_block = nn.Sequential(
            WSconv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSconv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSconv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def fade_in(self, alpha, downscaled, out):
        return (1 - alpha) * downscaled + alpha * out

    def minibatch_std(self, x):  ## NxCxHxW -> N -> 1 -> Nx1xHxW
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)  ## ex) 512 channels -> 513 channels in every batch sample

    ## very tricky
    def forward(self, x, alpha, steps):  ## steps=0 : 4x4 input / steps=1 : 8x8 input ...
        cur_step = len(self.progressive_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.last_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.progressive_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for i in range(cur_step + 1, len(self.progressive_blocks)):
            out = self.progressive_blocks[i](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.last_block(out).view(out.shape[0], -1)
