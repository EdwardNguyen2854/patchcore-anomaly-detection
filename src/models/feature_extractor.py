"""Pretrained backbone feature extractor for PatchCore."""

import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureExtractor(nn.Module):
    """Extract intermediate features from a pretrained WideResNet-50-2.

    Hooks into specified layers and returns their activations.

    Args:
        backbone: Model architecture name. Only 'wide_resnet50_2' supported.
        layers: List of layer names to extract features from.
        pretrained: Whether to load pretrained ImageNet weights.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list[str] | None = None,
        pretrained: bool = True,
    ):
        super().__init__()

        if backbone != "wide_resnet50_2":
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.layers = layers or ["layer2", "layer3"]

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = wide_resnet50_2(weights=weights)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Register hooks to capture intermediate features
        self._features: dict[str, torch.Tensor] = {}
        for layer_name in self.layers:
            layer = dict(self.model.named_children())[layer_name]
            layer.register_forward_hook(self._make_hook(layer_name))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from specified layers.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            List of feature tensors, one per layer.
        """
        self._features.clear()
        self.model(x)
        return [self._features[layer] for layer in self.layers]
