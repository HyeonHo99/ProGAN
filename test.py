import torch
from models import Generator, Discriminator
from math import log2

def test():
    Z_DIM = 100
    IN_CHANNELS = 128
    IMG_CHANNELS = 3
    gen = Generator(Z_DIM, IN_CHANNELS, IMG_CHANNELS)
    critic = Discriminator(IN_CHANNELS, IMG_CHANNELS)

    for img_size in [4, 8, 16, 32, 64, 128]:
        num_steps = int(log2(img_size / 4))
        z = torch.randn((1, Z_DIM, 1, 1))
        generated = gen(z, 0.5, steps=num_steps)

        assert generated.shape == (1, IMG_CHANNELS, img_size, img_size)

        critic_generated = critic(generated, 0.5, steps=num_steps)

        assert critic_generated.shape == (1, 1)
        print(f"Succes at image size {img_size}x{img_size}")

test()