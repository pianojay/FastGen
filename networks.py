"""U-Net backbones and EDM preconditioning.

Adapted from the official EDM implementation (Karras et al., "Elucidating the
Design Space of Diffusion-Based Generative Models", NeurIPS 2022,
https://github.com/NVlabs/edm, CC BY-NC-SA 4.0).  Changes: removed the
pickling/persistence machinery, replaced the custom attention op with
``scaled_dot_product_attention``, accept integer class labels, and optionally
return the encoder-bottleneck logits used as the discriminator in Adversarial
Score identity Distillation (Zhou et al., ICLR 2025, https://github.com/mingyuanzhou/SiD).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=(1, 1), fused_resample=False, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if self.fused_resample and self.up and w is not None:
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = F.conv2d(x, w, padding=w_pad + f_pad)
            x = F.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = F.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x + b.reshape(1, -1, 1, 1)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=(1, 1), resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels, out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(emb_channels, out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(out_channels, out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels, out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(out_channels, out_channels * 3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(out_channels, out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x + params))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            n, c, h, w = x.shape
            q, k, v = self.qkv(self.norm2(x)).reshape(n * self.num_heads, c // self.num_heads, 3, h * w).unbind(2)
            a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            x = self.proj(a.transpose(1, 2).reshape(n, c, h, w)) + x
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class SongUNet(nn.Module):
    """DDPM++ backbone (EDM reimplementation of Song et al., ICLR 2021)."""

    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0, augment_dim=0,
                 model_channels=128, channel_mult=(1, 2, 2, 2), channel_mult_emb=4, num_blocks=4,
                 attn_resolutions=(16,), dropout=0.10, label_dropout=0,
                 channel_mult_noise=1, resample_filter=(1, 1)):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                            init=init, init_zero=init_zero, init_attn=init_attn)

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_label = Linear(label_dim, noise_channels, **init) if label_dim else None
        self.map_augment = Linear(augment_dim, noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(noise_channels, emb_channels, **init)
        self.map_layer1 = Linear(emb_channels, emb_channels, **init)

        self.enc = nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(cin, cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(cout, cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]
        self.bottleneck_channels = cout

        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(cout, cout, attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(cout, cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(cout, cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=attn, **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout, eps=1e-6)
        self.out_conv = Conv2d(cout, out_channels, kernel=3, **init_zero)

    def embed(self, noise_labels, class_labels, augment_labels, batch):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([batch, 1], device=tmp.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        return F.silu(self.map_layer1(emb))

    def forward(self, x, noise_labels, class_labels, augment_labels=None, mode="decoder"):
        """mode: 'decoder' -> output, 'encoder' -> bottleneck features, 'both' -> (output, features).
        'encoder_ms' / 'both_ms': features = list with the last encoder activation of every resolution
        (training-time discriminator features only; sampling always uses 'decoder')."""
        emb = self.embed(noise_labels, class_labels, augment_labels, x.shape[0])
        skips = []
        last_per_res = {}
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
            last_per_res[x.shape[-1]] = x
        if mode in ("encoder_ms", "both_ms"):
            feats = list(last_per_res.values())
        else:
            feats = x if mode in ("encoder", "both") else None
        if mode in ("encoder", "encoder_ms"):
            return feats
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        out = self.out_conv(F.silu(self.out_norm(x)))
        return (out, feats) if mode in ("both", "both_ms") else out


class EDMPrecond(nn.Module):
    """EDM preconditioning: D(x; sigma) = c_skip x + c_out F(c_in x; c_noise)."""

    def __init__(self, img_resolution=64, img_channels=3, label_dim=0, sigma_data=0.5, **model_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.model = SongUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels,
                              label_dim=label_dim, **model_kwargs)

    def labels_to_onehot(self, class_labels, batch, device):
        if self.label_dim == 0:
            return None
        if class_labels is None:
            return torch.zeros([batch, self.label_dim], device=device)
        if class_labels.dtype in (torch.int64, torch.int32) and class_labels.dim() == 1:
            return F.one_hot(class_labels, self.label_dim).to(torch.float32)
        return class_labels.to(torch.float32).reshape(-1, self.label_dim)

    def forward(self, x, sigma, class_labels=None, augment_labels=None, mode="decoder"):
        x = x.to(torch.float32)
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device)
        sigma = sigma.reshape(-1, 1, 1, 1).expand(x.shape[0], 1, 1, 1)
        class_labels = self.labels_to_onehot(class_labels, x.shape[0], x.device)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        out = self.model(c_in * x, c_noise.flatten(), class_labels, augment_labels=augment_labels, mode=mode)
        if mode == "encoder":
            return out.to(torch.float32)
        if mode == "encoder_ms":
            return [f.to(torch.float32) for f in out]
        F_x, feats = out if mode in ("both", "both_ms") else (out, None)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        if mode == "both":
            return D_x, feats.to(torch.float32)
        if mode == "both_ms":
            return D_x, [f.to(torch.float32) for f in feats]
        return D_x
