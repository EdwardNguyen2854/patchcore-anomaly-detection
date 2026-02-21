"""Evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute image-level AUROC.

    Args:
        labels: Binary labels (0=normal, 1=anomalous), shape (N,).
        scores: Anomaly scores, shape (N,).

    Returns:
        AUROC score.
    """
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(masks: np.ndarray, score_maps: np.ndarray) -> float:
    """Compute pixel-level AUROC.

    Args:
        masks: Binary ground truth masks, shape (N, H, W).
        score_maps: Pixel-level anomaly score maps, shape (N, H, W).

    Returns:
        Pixel-level AUROC score.
    """
    return float(roc_auc_score(masks.ravel(), score_maps.ravel()))


def compute_pro(masks: np.ndarray, score_maps: np.ndarray, num_thresholds: int = 200) -> float:
    """Compute Per-Region Overlap (PRO) metric.

    Averages the per-connected-component overlap at each threshold,
    then computes AUC up to FPR=0.3.

    Args:
        masks: Binary ground truth masks, shape (N, H, W).
        score_maps: Pixel-level anomaly score maps, shape (N, H, W).
        num_thresholds: Number of thresholds to evaluate.

    Returns:
        PRO AUC score (normalized to [0, 1]).
    """
    from scipy.ndimage import label as label_connected

    flat_masks = masks.ravel().astype(bool)
    flat_scores = score_maps.ravel()

    # Compute FPR values at different thresholds
    thresholds = np.linspace(flat_scores.max(), flat_scores.min(), num_thresholds)

    fprs = []
    pros = []

    for threshold in thresholds:
        binary_pred = flat_scores >= threshold
        fp = np.sum(binary_pred & ~flat_masks)
        tn = np.sum(~binary_pred & ~flat_masks)
        fpr = fp / (fp + tn + 1e-8)

        # Per-region overlap
        region_overlaps = []
        for i in range(masks.shape[0]):
            labeled, num_regions = label_connected(masks[i])
            pred_binary = score_maps[i] >= threshold
            for region_id in range(1, num_regions + 1):
                region_mask = labeled == region_id
                overlap = np.sum(pred_binary & region_mask) / np.sum(region_mask)
                region_overlaps.append(overlap)

        pro = np.mean(region_overlaps) if region_overlaps else 0.0
        fprs.append(fpr)
        pros.append(pro)

    fprs = np.array(fprs)
    pros = np.array(pros)

    # Sort by FPR
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    pros = pros[sorted_idx]

    # AUC up to FPR=0.3, normalized
    valid = fprs <= 0.3
    if valid.sum() < 2:
        return 0.0
    return float(auc(fprs[valid], pros[valid]) / 0.3)


def compute_optimal_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute optimal F1 score and its threshold.

    Args:
        labels: Binary labels (0=normal, 1=anomalous), shape (N,).
        scores: Anomaly scores, shape (N,).

    Returns:
        Tuple of (best_f1, threshold).
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
    return best_f1, best_threshold


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray | None = None,
    score_maps: np.ndarray | None = None,
) -> dict:
    """Compute all anomaly detection metrics.

    Args:
        labels: Binary image-level labels, shape (N,).
        scores: Image-level anomaly scores, shape (N,).
        masks: Optional pixel-level ground truth masks, shape (N, H, W).
        score_maps: Optional pixel-level score maps, shape (N, H, W).

    Returns:
        Dict with metric names and values.
    """
    results = {}

    results["image_auroc"] = compute_image_auroc(labels, scores)
    f1, threshold = compute_optimal_f1(labels, scores)
    results["image_f1"] = f1
    results["image_threshold"] = threshold

    if masks is not None and score_maps is not None:
        # Only compute pixel metrics if there are anomalous pixels
        if masks.sum() > 0:
            results["pixel_auroc"] = compute_pixel_auroc(masks, score_maps)
            results["pixel_pro"] = compute_pro(masks, score_maps)
        else:
            results["pixel_auroc"] = None
            results["pixel_pro"] = None

    return results
