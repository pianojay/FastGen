"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import torch
from torch import nn

from src.utils import count_parameters, parameter_summary


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()


# ---------------------------------------------------------------------------
# Student models: an EDM-preconditioned DDPM++ denoiser (networks.py) distilled
# into a few-step generator with adversarial Score identity Distillation
# (train_sida.py).  Step k feeds x_{k-1} + sigma_k * z_k to the denoiser at the
# fixed noise level sigma_k (x_{-1} = 0), i.e. exactly one backbone evaluation
# per step.

NUM_CLASSES = 151
# augment_dim must match the teacher the student was distilled from (0 = teacher trained
# without augmentation conditioning); it is unused at sampling time (no augmentation).
ONE_NFE_CONFIG = dict(
    img_resolution=64, img_channels=3, label_dim=NUM_CLASSES, augment_dim=0,
    model_channels=128, channel_mult=(1, 2, 2, 2), num_blocks=4, attn_resolutions=(16,),
    dropout=0.0,
)
FEW_NFE_CONFIG = dict(ONE_NFE_CONFIG, model_channels=160)  # 96.6M parameters
ONE_NFE_SIGMAS = (2.5,)
FEW_NFE_SIGMAS = (2.5, 0.4)
SAMPLE_SEED = 0


def _build_backbone(config):
    from networks import EDMPrecond

    return EDMPrecond(**config)


class _FewStepGenerator(Model):
    def __init__(self, sigmas, config, device="cpu"):
        super().__init__(_build_backbone(config))
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self._generator = None
        self._device = device

    @torch.no_grad()
    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        device = torch.device(device)
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device).manual_seed(SAMPLE_SEED)
        if category is None:
            category = torch.zeros(shape[0], dtype=torch.long, device=device)
        category = category.to(device).long()
        x = torch.zeros(shape, device=device)
        for k, sigma in enumerate(self.sigmas.tolist()):
            z = torch.randn(shape, device=device, generator=self._generator)
            x = self.backbone(x + sigma * z if k else sigma * z,
                              torch.full((shape[0],), sigma, device=device), category)
        return x.clamp(-1, 1)


class ModelOneNFE(_FewStepGenerator):
    """Student model evaluated with exactly one model function evaluation."""

    def __init__(self, device="cpu", **kwargs):
        super().__init__(ONE_NFE_SIGMAS, ONE_NFE_CONFIG, device=device)


class ModelFewNFE(_FewStepGenerator):
    """Student model evaluated with no more than four model evaluations."""

    def __init__(self, device="cpu", **kwargs):
        super().__init__(FEW_NFE_SIGMAS, FEW_NFE_CONFIG, device=device)
