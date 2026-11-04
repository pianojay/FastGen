"""One deterministic class-conditioned U-Net and a scratch projection critic."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .unet import SongUNet


class Generator(nn.Module):
    def __init__(self, model_channels=128, num_blocks=2):
        super().__init__()
        self.network = SongUNet(
            img_resolution=64, in_channels=3, out_channels=3, label_dim=151,
            augment_dim=0, model_channels=model_channels, channel_mult=[1, 2, 2, 2],
            channel_mult_emb=4, num_blocks=num_blocks, attn_resolutions=[16],
            dropout=0, label_dropout=0, embedding_type='positional',
            channel_mult_noise=2, encoder_type='standard', decoder_type='standard',
            resample_filter=[1, 3, 3, 1])

    def forward(self, noise, category):
        return self.network(noise, category)


class DownBlock(nn.Module):
    def __init__(self, cin, cout, down=True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1)
        self.down = down

    def forward(self, x):
        h = self.conv2(F.leaky_relu(self.conv1(F.leaky_relu(x, 0.2)), 0.2))
        skip = self.skip(x)
        if self.down:
            h, skip = F.avg_pool2d(h, 2), F.avg_pool2d(skip, 2)
        return (h + skip) / math.sqrt(2)


class Discriminator(nn.Module):
    """Residual 64->4 spatial critic; no batch statistics or pretrained features."""
    def __init__(self, channels=64):
        super().__init__()
        widths = [3, channels, 2*channels, 4*channels, 8*channels]
        self.blocks = nn.Sequential(*(DownBlock(a, b) for a, b in zip(widths, widths[1:])),
                                    DownBlock(widths[-1], widths[-1], down=False))
        self.head = nn.Linear(widths[-1], 1)
        self.embedding = nn.Embedding(151, widths[-1])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, std=1 / math.sqrt(widths[-1]))

    def forward(self, image, category):
        h = F.leaky_relu(self.blocks(image), 0.2).mean((2, 3))
        return self.head(h).flatten() + (h * self.embedding(category)).sum(1)
