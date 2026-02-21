"""Visualization utilities for anomaly detection results."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to uint8 numpy array.

    Args:
        image: Tensor of shape (3, H, W) with ImageNet normalization.

    Returns:
        Numpy array of shape (H, W, 3) in uint8 [0, 255].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def create_heatmap_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay an anomaly heatmap on the original image.

    Args:
        image: Original image (H, W, 3) in uint8.
        score_map: Anomaly score map (H, W), any range.
        alpha: Blend factor for the heatmap overlay.

    Returns:
        Blended image (H, W, 3) in uint8.
    """
    # Normalize score map to [0, 255]
    smin, smax = score_map.min(), score_map.max()
    if smax - smin > 1e-8:
        normalized = ((score_map - smin) / (smax - smin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(score_map, dtype=np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def visualize_predictions(
    images: torch.Tensor,
    score_maps: torch.Tensor,
    masks: torch.Tensor | None = None,
    image_scores: torch.Tensor | None = None,
    labels: list[int] | None = None,
    save_path: str | None = None,
    max_images: int = 8,
) -> plt.Figure:
    """Create a grid visualization of anomaly detection results.

    Shows: original image | heatmap overlay | ground truth mask (if available).

    Args:
        images: Batch of images (B, 3, H, W).
        score_maps: Anomaly score maps (B, H, W).
        masks: Optional ground truth masks (B, 1, H, W).
        image_scores: Optional image-level scores (B,).
        labels: Optional list of labels (0=normal, 1=anomalous).
        save_path: Optional path to save the figure.
        max_images: Maximum number of images to display.

    Returns:
        Matplotlib figure.
    """
    n = min(len(images), max_images)
    has_masks = masks is not None
    n_cols = 3 if has_masks else 2

    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img = denormalize(images[i])
        smap = score_maps[i].cpu().numpy()

        # Original image
        axes[i, 0].imshow(img)
        title = "Original"
        if labels is not None:
            title += f" ({'Anomalous' if labels[i] else 'Normal'})"
        axes[i, 0].set_title(title)
        axes[i, 0].axis("off")

        # Heatmap overlay
        overlay = create_heatmap_overlay(img, smap)
        axes[i, 1].imshow(overlay)
        score_text = ""
        if image_scores is not None:
            score_text = f" (score: {image_scores[i]:.2f})"
        axes[i, 1].set_title(f"Anomaly Map{score_text}")
        axes[i, 1].axis("off")

        # Ground truth mask
        if has_masks:
            mask = masks[i].squeeze().cpu().numpy()
            axes[i, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    return fig
