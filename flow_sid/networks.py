"""Direct velocity wrapper; no EDM preconditioning, CFG, text or latent model."""
import torch
from .unet import SongUNet

ARCHITECTURE = dict(img_resolution=64, in_channels=3, out_channels=3,
    label_dim=151, augment_dim=0, model_channels=128, channel_mult=[1, 2, 2, 2],
    channel_mult_emb=4, num_blocks=4, attn_resolutions=[16], dropout=0.25,
    label_dropout=0, embedding_type='positional', channel_mult_noise=1,
    encoder_type='standard', decoder_type='standard', resample_filter=[1, 1])


class FlowNet(torch.nn.Module):
    def __init__(self, checkpointing=False, architecture=None):
        super().__init__()
        self.backbone = SongUNet(**(architecture or ARCHITECTURE))
        self.backbone.gradient_checkpointing = checkpointing

    def forward(self, x, t, labels, return_flag='decoder'):
        if labels.shape != (x.shape[0],) or labels.dtype != torch.long:
            raise ValueError('One int64 class ID is required per image')
        t = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        classes = torch.nn.functional.one_hot(labels, 151).float()
        result = self.backbone(x, t * 1000, classes, return_flag=return_flag)
        if isinstance(result, tuple):
            return tuple(v.float() for v in result)
        return result.float()


def parameter_count(net):
    count = sum(p.numel() for p in net.parameters())
    if count > 100_000_000:
        raise RuntimeError(f'Full framework exceeds 100M: {count}')
    return count

