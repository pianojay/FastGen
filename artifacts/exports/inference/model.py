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





class ModelOneNFE(Model):
    """Student model evaluated with exactly one model function evaluation."""

    def __init__(self, device="cpu", **kwargs):
        from flow_sid.networks import FlowNet
        super().__init__(FlowNet())
        self.register_buffer("sampler_nfe", torch.tensor(1, dtype=torch.long))
        self.register_buffer("sampler_ladder", torch.tensor([1., 0., 0., 0.]))
        self.to(device)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return _sample_student(self, shape, device, category, kwargs, one_nfe=True)


class ModelFewNFE(Model):
    """Student model evaluated with no more than four model evaluations."""

    def __init__(self, device="cpu", **kwargs):
        from flow_sid.networks import FlowNet
        super().__init__(FlowNet())
        self.register_buffer("sampler_nfe", torch.tensor(4, dtype=torch.long))
        self.register_buffer("sampler_ladder", torch.tensor([1., .75, .5, .25]))
        self.to(device)
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return _sample_student(self, shape, device, category, kwargs, one_nfe=False)


@torch.no_grad()
def _sample_student(model, shape, device, category, kwargs, one_nfe):
    from flow_sid.flow import ladder, student_rollout
    nfe = int(model.sampler_nfe.item())
    if nfe not in ([1] if one_nfe else [2, 3, 4]):
        raise ValueError("Invalid exported sampler NFE")
    if tuple(shape[1:]) != (3, 64, 64) or category is None:
        raise ValueError("Expected class-conditioned [B,3,64,64] samples")
    expected = torch.tensor(list(ladder(nfe)), device=model.sampler_ladder.device)
    if not torch.equal(model.sampler_ladder[:nfe], expected):
        raise ValueError("Sampler metadata does not match its declared ladder")
    device = torch.device(device)
    category = category.to(device=device, dtype=torch.long)
    if category.shape != (shape[0],) or ((category < 0) | (category >= 151)).any():
        raise ValueError("Category IDs must be in [0,150]")
    generator = kwargs.get("generator")
    noises = kwargs.get("noises")
    if noises is None:
        noises = [torch.randn(shape, device=device, generator=generator) for _ in range(nfe)]
    model.eval()
    with torch.autocast(device.type, dtype=torch.float16,
                        enabled=device.type == "cuda" and kwargs.get("amp", True)):
        return student_rollout(model.backbone, noises, category, nfe)
