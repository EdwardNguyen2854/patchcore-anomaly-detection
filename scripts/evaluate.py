"""Evaluation entrypoint for PatchCore anomaly detection."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CATEGORIES, MVTecADDataset
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualize import visualize_predictions
from src.models.patchcore import PatchCore
from src.utils.config import load_config


def evaluate_category(config: dict, category: str) -> dict:
    """Evaluate PatchCore on a single category.

    Args:
        config: Configuration dict.
        category: MVTec AD category name.

    Returns:
        Dict with evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating category: {category}")
    print(f"{'='*60}")

    output_dir = Path(config["output"]["dir"]) / category
    model_path = output_dir / "model.pt"

    if not model_path.exists():
        print(f"Model not found at {model_path}. Run training first.")
        return {}

    # Load model (neighborhood_size and num_neighbors are restored from the saved state)
    model = PatchCore(
        backbone=config["model"]["backbone"],
        layers=config["model"]["layers"],
        pretrained=config["model"]["pretrained"],
        device=config["device"],
    )
    model.load(str(model_path))

    # Load test data
    dataset = MVTecADDataset(
        root=config["dataset"]["root"],
        category=category,
        split="test",
        image_size=config["dataset"]["image_size"],
        center_crop=config["dataset"]["center_crop"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        pin_memory=True,
    )
    print(f"Test samples: {len(dataset)}")

    # Run inference
    all_labels = []
    all_scores = []
    all_masks = []
    all_score_maps = []
    all_images = []

    for batch in tqdm(dataloader, desc="Inference"):
        images = batch["image"]
        image_scores, score_maps = model.predict(images)

        all_labels.extend(batch["label"].numpy().tolist())
        all_scores.append(image_scores.cpu().numpy())
        all_masks.append(batch["mask"].numpy())
        all_score_maps.append(score_maps.cpu().numpy())
        all_images.append(images)

    labels = np.array(all_labels)
    scores = np.concatenate(all_scores)
    masks = np.concatenate(all_masks).squeeze(1)  # (N, H, W)
    score_maps = np.concatenate(all_score_maps)

    # Compute metrics
    metrics = compute_metrics(labels, scores, masks, score_maps)
    metrics["category"] = category
    metrics["num_test_samples"] = len(dataset)

    print(f"  Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"  Image F1:    {metrics['image_f1']:.4f}")
    if metrics.get("pixel_auroc") is not None:
        print(f"  Pixel AUROC: {metrics['pixel_auroc']:.4f}")
        print(f"  Pixel PRO:   {metrics['pixel_pro']:.4f}")

    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save visualizations
    if config["output"]["save_visualizations"]:
        all_images_cat = torch.cat(all_images)
        # Pick some anomalous samples to visualize
        anomalous_idx = np.where(labels == 1)[0]
        if len(anomalous_idx) > 0:
            vis_idx = anomalous_idx[:8]
            vis_images = all_images_cat[vis_idx]
            vis_score_maps = torch.from_numpy(score_maps[vis_idx])
            vis_masks = torch.from_numpy(masks[vis_idx]).unsqueeze(1)
            vis_scores = torch.from_numpy(scores[vis_idx])
            vis_labels = labels[vis_idx].tolist()

            vis_path = output_dir / "visualizations.png"
            visualize_predictions(
                vis_images, vis_score_maps, vis_masks, vis_scores, vis_labels,
                save_path=str(vis_path),
            )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore on MVTec AD")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--category", type=str, default=None, help="Category to evaluate")
    args = parser.parse_args()

    config = load_config(args.config)
    category = args.category or config["dataset"]["category"]

    if category == "all":
        categories = CATEGORIES
    else:
        categories = [category]

    all_results = []
    for cat in categories:
        metrics = evaluate_category(config, cat)
        if metrics:
            all_results.append(metrics)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Overall Results")
        print(f"{'='*60}")
        print(f"{'Category':15s} | {'Img AUROC':>9s} | {'Pxl AUROC':>9s} | {'Img F1':>6s}")
        print("-" * 50)
        for r in all_results:
            pxl = f"{r['pixel_auroc']:.4f}" if r.get('pixel_auroc') is not None else "   N/A"
            print(f"{r['category']:15s} | {r['image_auroc']:9.4f} | {pxl:>9s} | {r['image_f1']:.4f}")

        # Save combined results
        combined_path = Path(config["output"]["dir"]) / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
