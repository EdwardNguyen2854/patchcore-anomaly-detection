"""PatchCore anomaly detection model.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm

from src.models.feature_extractor import FeatureExtractor


class PatchCore:
    """PatchCore anomaly detection model.

    Args:
        backbone: Feature extractor backbone name.
        layers: Backbone layers to extract features from.
        pretrained: Use pretrained backbone weights.
        neighborhood_size: Kernel size for locally-aware patch aggregation (paper §3.1).
            Each patch feature is averaged with its spatial neighbours before building
            the memory bank.  Must be odd.  Use 1 to disable.
        coreset_ratio: Fraction of patches to keep via coreset subsampling.
        num_neighbors: Number of nearest neighbors for anomaly scoring.
        device: Device to run on.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list[str] | None = None,
        pretrained: bool = True,
        neighborhood_size: int = 3,
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cpu",
    ):
        self.neighborhood_size = neighborhood_size
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.device = device

        self.feature_extractor = FeatureExtractor(
            backbone=backbone, layers=layers, pretrained=pretrained
        ).to(device)

        self.memory_bank: torch.Tensor | None = None
        self._feature_map_dims: tuple[int, int] | None = None

    def _embed_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Combine multi-scale features into locally-aware patch embeddings.

        For each feature map:
          1. Upsample to the largest spatial resolution.
          2. Apply average pooling over a ``neighborhood_size × neighborhood_size``
             window (stride=1, same padding) so every patch aggregates its spatial
             neighbours — the "locally aware patch features" described in §3.1.
          3. Concatenate along the channel dimension and reshape to (N_patches, C).

        Args:
            features: List of feature tensors from different backbone layers.

        Returns:
            Patch embeddings of shape (B * H * W, C).
        """
        target_size = features[0].shape[2:]

        embeddings = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)

            # Locally aware aggregation: avg-pool over a neighbourhood.
            # padding = neighborhood_size // 2 keeps spatial dimensions unchanged.
            if self.neighborhood_size > 1:
                feat = F.avg_pool2d(
                    feat,
                    kernel_size=self.neighborhood_size,
                    stride=1,
                    padding=self.neighborhood_size // 2,
                )
            embeddings.append(feat)

        # (B, C_total, H, W)
        combined = torch.cat(embeddings, dim=1)
        B, C, H, W = combined.shape
        self._feature_map_dims = (H, W)

        # (B * H * W, C)
        return combined.permute(0, 2, 3, 1).reshape(-1, C)

    def _coreset_subsample(self, embeddings: torch.Tensor, seed: int = 42) -> torch.Tensor:
        """Greedy coreset subsampling to reduce memory bank size.

        Uses sparse random projection for speed, then iteratively selects
        the point farthest from the current coreset.

        Args:
            embeddings: Patch embeddings of shape (N, C).
            seed: Random seed for reproducibility.

        Returns:
            Subsampled memory bank of shape (M, C) where M = N * coreset_ratio.
        """
        n_samples = embeddings.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_ratio))

        if n_coreset >= n_samples:
            return embeddings

        rng = np.random.RandomState(seed)

        # Project to lower dimension for faster distance computation
        embeddings_np = embeddings.cpu().numpy()
        projector = SparseRandomProjection(n_components="auto", eps=0.9, random_state=rng)
        projected = projector.fit_transform(embeddings_np)

        # Greedy coreset: start from a random point, always pick the farthest
        selected_indices = [rng.randint(n_samples)]
        min_distances = np.full(n_samples, np.inf)

        for _ in tqdm(range(n_coreset - 1), desc="Coreset subsampling", leave=False):
            last = projected[selected_indices[-1]]
            distances = np.linalg.norm(projected - last, axis=1)
            min_distances = np.minimum(min_distances, distances)
            selected_indices.append(int(np.argmax(min_distances)))

        return embeddings[selected_indices]

    def fit(self, dataloader: torch.utils.data.DataLoader, seed: int = 42) -> None:
        """Build the memory bank from training data.

        Args:
            dataloader: DataLoader yielding dicts with 'image' tensors.
            seed: Random seed for coreset subsampling.
        """
        self.feature_extractor.eval()
        all_embeddings = []

        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(self.device)
            features = self.feature_extractor(images)
            embeddings = self._embed_features(features)
            all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Total patch embeddings: {all_embeddings.shape[0]:,} (dim={all_embeddings.shape[1]})")

        self.memory_bank = self._coreset_subsample(all_embeddings, seed=seed).to(self.device)
        print(f"Memory bank size after coreset: {self.memory_bank.shape[0]:,}")

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly scores for a batch of images.

        Patch scores use the score re-weighting scheme from §3.2 of the paper:

            patch_score = (1 − w) × d₁
            w = softmax(−[d₁, d₂, …, dₖ])[0]

        where d₁ ≤ d₂ ≤ … ≤ dₖ are the k-NN distances to the memory bank.

        The weight w is the softmax weight of the *nearest* neighbour over the k
        distances using negative distances as logits.  For a normal patch the
        nearest neighbour is much closer than the rest, so w → 1 and the score
        is strongly suppressed.  For an anomalous patch all k neighbours are far
        and similarly spaced, so w ≈ 1/k and the score is preserved.

        When num_neighbors == 1 the re-weighting is skipped (softmax of a single
        element is always 1, giving a zero score for everything).

        Args:
            images: Input tensor of shape (B, 3, H, W).

        Returns:
            Tuple of:
                - image_scores: (B,) image-level anomaly scores
                - score_maps: (B, H, W) pixel-level anomaly score maps
                  (upsampled to input image spatial size)
        """
        if self.memory_bank is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        self.feature_extractor.eval()
        images = images.to(self.device)

        features = self.feature_extractor(images)
        embeddings = self._embed_features(features)

        B = images.shape[0]
        H, W = self._feature_map_dims
        img_H, img_W = images.shape[2], images.shape[3]

        # Compute pairwise distances to memory bank
        # embeddings: (B*H*W, C), memory_bank: (M, C)
        distances = torch.cdist(embeddings, self.memory_bank, p=2)  # (B*H*W, M)

        # k-nearest neighbor distances, ascending
        knn_dist, _ = distances.topk(self.num_neighbors, largest=False, dim=1)  # (B*H*W, k)

        # Score re-weighting (paper §3.2).
        # Softmax over *negative* k-NN distances: the nearest neighbour (smallest d)
        # receives the largest logit (least negative) and therefore the highest
        # softmax weight w.
        #   • Normal patch  (d₁ ≪ d₂,…,dₖ): w → 1  →  (1−w) → 0  →  score ≈ 0
        #   • Anomalous patch (d₁ ≈ d₂ ≈…≈ dₖ): w ≈ 1/k  →  (1−w) ≈ (k−1)/k
        # Skip re-weighting when k=1: softmax of a single value is always 1.0.
        if self.num_neighbors > 1:
            softmax_weights = F.softmax(-knn_dist, dim=1)   # (B*H*W, k)
            reweight = 1.0 - softmax_weights[:, 0]          # (B*H*W,)
            patch_scores = reweight * knn_dist[:, 0]        # (B*H*W,)
        else:
            patch_scores = knn_dist[:, 0]                   # (B*H*W,)

        score_maps = patch_scores.reshape(B, H, W)

        # Upsample to original image size
        score_maps = F.interpolate(
            score_maps.unsqueeze(1), size=(img_H, img_W), mode="bilinear", align_corners=False
        ).squeeze(1)

        # Apply Gaussian smoothing
        score_maps_np = score_maps.cpu().numpy()
        for i in range(B):
            score_maps_np[i] = gaussian_filter(score_maps_np[i], sigma=4)
        score_maps = torch.from_numpy(score_maps_np).to(self.device)

        # Image-level score = max pixel score
        image_scores = score_maps.reshape(B, -1).max(dim=1).values

        return image_scores, score_maps

    def save(self, path: str) -> None:
        """Save the memory bank and model config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "feature_map_dims": self._feature_map_dims,
            "neighborhood_size": self.neighborhood_size,
            "coreset_ratio": self.coreset_ratio,
            "num_neighbors": self.num_neighbors,
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load memory bank from disk.

        ``neighborhood_size`` defaults to 1 (no aggregation) when loading a
        model saved by an older version of this code, preserving the behaviour
        under which that model was trained.
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.memory_bank = state["memory_bank"].to(self.device)
        self._feature_map_dims = state["feature_map_dims"]
        self.neighborhood_size = state.get("neighborhood_size", 1)
        self.coreset_ratio = state["coreset_ratio"]
        self.num_neighbors = state["num_neighbors"]
        print(f"Model loaded from {path} (memory bank: {self.memory_bank.shape[0]:,} patches)")
