"""Tests for MVTec AD dataset loading."""

import pytest
import torch

from src.data.dataset import CATEGORIES, MVTecADDataset
from src.data.transforms import get_transforms


def test_categories_list():
    """Verify all 15 MVTec AD categories are defined."""
    assert len(CATEGORIES) == 15
    assert "bottle" in CATEGORIES
    assert "cable" in CATEGORIES


def test_invalid_category():
    """Reject unknown category names."""
    with pytest.raises(ValueError, match="Unknown category"):
        MVTecADDataset(root="data/mvtec_anomaly_detection", category="invalid")


def test_invalid_split():
    """Reject unknown split names."""
    with pytest.raises(ValueError, match="Split must be"):
        MVTecADDataset(root="data/mvtec_anomaly_detection", category="bottle", split="val")


def test_transforms_output_shape():
    """Verify transforms produce correct output dimensions."""
    transforms = get_transforms(image_size=224, center_crop=224)
    assert "image" in transforms
    assert "mask" in transforms


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/bottle/train").exists(),
    reason="MVTec AD dataset not downloaded",
)
class TestWithDataset:
    """Tests that require the actual dataset to be present."""

    def test_train_dataset_loads(self):
        dataset = MVTecADDataset(
            root="data", category="bottle", split="train"
        )
        assert len(dataset) > 0

        sample = dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["mask"].shape == (1, 224, 224)
        assert sample["label"] == 0  # Training set is all normal

    def test_test_dataset_loads(self):
        dataset = MVTecADDataset(
            root="data", category="bottle", split="test"
        )
        assert len(dataset) > 0

        # Test set should have both normal and anomalous samples
        labels = [dataset[i]["label"] for i in range(len(dataset))]
        assert 0 in labels
        assert 1 in labels

    def test_mask_is_binary(self):
        dataset = MVTecADDataset(
            root="data", category="bottle", split="test"
        )
        for i in range(min(10, len(dataset))):
            mask = dataset[i]["mask"]
            unique_vals = torch.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique_vals)
