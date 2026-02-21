"""Tests for PatchCore model and feature extraction."""

import pytest
import torch

from src.models.feature_extractor import FeatureExtractor
from src.models.patchcore import PatchCore


class TestFeatureExtractor:
    """Tests for the feature extraction backbone."""

    def test_output_layers(self):
        """Feature extractor returns features from specified layers."""
        extractor = FeatureExtractor(pretrained=False, layers=["layer2", "layer3"])
        dummy = torch.randn(1, 3, 224, 224)
        features = extractor(dummy)

        assert len(features) == 2
        # layer2 output: (B, 512, 28, 28) for WRN-50-2
        assert features[0].shape[0] == 1
        assert features[0].shape[1] == 512
        # layer3 output: (B, 1024, 14, 14) for WRN-50-2
        assert features[1].shape[0] == 1
        assert features[1].shape[1] == 1024

    def test_frozen_weights(self):
        """Backbone weights should be frozen."""
        extractor = FeatureExtractor(pretrained=False)
        for param in extractor.model.parameters():
            assert not param.requires_grad

    def test_invalid_backbone(self):
        """Reject unsupported backbone architectures."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            FeatureExtractor(backbone="resnet18")


class TestPatchCore:
    """Tests for the PatchCore model."""

    def test_embed_features(self):
        """Feature embedding combines multi-scale features correctly."""
        model = PatchCore(pretrained=False, device="cpu")
        # Simulate two feature maps from different layers
        feat1 = torch.randn(2, 512, 28, 28)
        feat2 = torch.randn(2, 1024, 14, 14)

        embeddings = model._embed_features([feat1, feat2])
        # Should be (B * H * W, C_total) where H, W = 28, 28
        assert embeddings.shape == (2 * 28 * 28, 512 + 1024)

    def test_predict_without_fit_raises(self):
        """Calling predict before fit should raise an error."""
        model = PatchCore(pretrained=False, device="cpu")
        dummy = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(dummy)

    def test_fit_and_predict(self):
        """End-to-end test: fit on dummy data, then predict."""
        model = PatchCore(
            pretrained=False,
            coreset_ratio=0.5,  # High ratio for small test
            num_neighbors=1,
            device="cpu",
        )

        # Create a minimal dummy dataloader
        dummy_data = [
            {"image": torch.randn(2, 3, 224, 224)},
            {"image": torch.randn(2, 3, 224, 224)},
        ]
        model.fit(dummy_data)

        assert model.memory_bank is not None
        assert model.memory_bank.shape[0] > 0

        # Predict on a single image
        test_image = torch.randn(1, 3, 224, 224)
        image_scores, score_maps = model.predict(test_image)

        assert image_scores.shape == (1,)
        assert score_maps.shape == (1, 224, 224)
        assert image_scores[0].item() > 0

    def test_save_and_load(self, tmp_path):
        """Model can be saved and loaded correctly, including neighborhood_size."""
        model = PatchCore(pretrained=False, coreset_ratio=0.5, neighborhood_size=3, device="cpu")
        dummy_data = [{"image": torch.randn(2, 3, 224, 224)}]
        model.fit(dummy_data)

        save_path = str(tmp_path / "model.pt")
        model.save(save_path)

        model2 = PatchCore(pretrained=False, device="cpu")
        model2.load(save_path)

        assert model2.memory_bank is not None
        assert model2.memory_bank.shape == model.memory_bank.shape
        assert torch.allclose(model2.memory_bank, model.memory_bank)
        assert model2.neighborhood_size == model.neighborhood_size
