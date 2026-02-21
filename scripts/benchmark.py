"""Hyperparameter ablation study for PatchCore.

Sweeps coreset_ratio and num_neighbors to measure their impact on detection
performance. Results are saved to outputs/benchmark/ and printed as a table.

Usage:
    python scripts/benchmark.py --config configs/default.yaml --category bottle
    make benchmark CATEGORY=bottle
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import MVTecADDataset
from src.evaluation.metrics import compute_metrics
from src.models.patchcore import PatchCore
from src.utils.config import load_config


# ── Ablation grids ────────────────────────────────────────────────────────────

CORESET_RATIOS = [0.01, 0.05, 0.1, 0.25]
NUM_NEIGHBORS  = [1, 3, 5, 9, 15]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_dataloader(config: dict, category: str, split: str) -> torch.utils.data.DataLoader:
    dataset = MVTecADDataset(
        root=config["dataset"]["root"],
        category=category,
        split=split,
        image_size=config["dataset"]["image_size"],
        center_crop=config["dataset"]["center_crop"],
    )
    batch_key = "train" if split == "train" else "eval"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config[batch_key]["batch_size"],
        shuffle=False,
        num_workers=config[batch_key]["num_workers"],
        pin_memory=True,
    ), len(dataset)


def train_and_evaluate(config: dict, category: str) -> dict:
    """Train a PatchCore model and return evaluation metrics + timing."""
    train_loader, n_train = build_dataloader(config, category, "train")
    test_loader,  _       = build_dataloader(config, category, "test")

    model = PatchCore(
        backbone=config["model"]["backbone"],
        layers=config["model"]["layers"],
        pretrained=config["model"]["pretrained"],
        neighborhood_size=config["model"].get("neighborhood_size", 3),
        coreset_ratio=config["model"]["coreset_ratio"],
        num_neighbors=config["model"]["num_neighbors"],
        device=config["device"],
    )

    t0 = time.time()
    model.fit(train_loader, seed=config["train"]["seed"])
    train_time = time.time() - t0

    all_labels, all_scores, all_masks, all_score_maps = [], [], [], []

    t0 = time.time()
    for batch in tqdm(test_loader, desc="Inference", leave=False):
        images = batch["image"]
        image_scores, score_maps = model.predict(images)
        all_labels.extend(batch["label"].numpy().tolist())
        all_scores.append(image_scores.cpu().numpy())
        all_masks.append(batch["mask"].numpy())
        all_score_maps.append(score_maps.cpu().numpy())
    infer_time = time.time() - t0

    labels     = np.array(all_labels)
    scores     = np.concatenate(all_scores)
    masks      = np.concatenate(all_masks).squeeze(1)
    score_maps = np.concatenate(all_score_maps)

    metrics = compute_metrics(labels, scores, masks, score_maps)
    metrics["train_time_s"]  = round(train_time, 1)
    metrics["infer_time_s"]  = round(infer_time, 1)
    metrics["memory_bank_size"] = model.memory_bank.shape[0]
    return metrics


# ── Ablation runners ──────────────────────────────────────────────────────────

def run_coreset_ablation(base_config: dict, category: str) -> list[dict]:
    """Vary coreset_ratio, keep num_neighbors fixed at default."""
    results = []
    default_k = base_config["model"]["num_neighbors"]
    print(f"\n{'─'*60}")
    print(f"Ablation: coreset_ratio  (num_neighbors={default_k})")
    print(f"{'─'*60}")

    for ratio in CORESET_RATIOS:
        cfg = copy.deepcopy(base_config)
        cfg["model"]["coreset_ratio"] = ratio

        print(f"\n  coreset_ratio={ratio}")
        metrics = train_and_evaluate(cfg, category)
        results.append({
            "experiment": "coreset_ratio",
            "coreset_ratio": ratio,
            "num_neighbors": default_k,
            **metrics,
        })
        img_auroc = metrics["image_auroc"]
        pxl_auroc = metrics.get("pixel_auroc") or float("nan")
        print(f"    Image AUROC: {img_auroc:.4f}  |  Pixel AUROC: {pxl_auroc:.4f}"
              f"  |  Memory bank: {metrics['memory_bank_size']:,}"
              f"  |  Train: {metrics['train_time_s']}s")

    return results


def run_knn_ablation(base_config: dict, category: str) -> list[dict]:
    """Vary num_neighbors, keep coreset_ratio fixed at default."""
    results = []
    default_ratio = base_config["model"]["coreset_ratio"]
    print(f"\n{'─'*60}")
    print(f"Ablation: num_neighbors  (coreset_ratio={default_ratio})")
    print(f"{'─'*60}")

    for k in NUM_NEIGHBORS:
        cfg = copy.deepcopy(base_config)
        cfg["model"]["num_neighbors"] = k

        print(f"\n  num_neighbors={k}")
        metrics = train_and_evaluate(cfg, category)
        results.append({
            "experiment": "num_neighbors",
            "coreset_ratio": default_ratio,
            "num_neighbors": k,
            **metrics,
        })
        img_auroc = metrics["image_auroc"]
        pxl_auroc = metrics.get("pixel_auroc") or float("nan")
        print(f"    Image AUROC: {img_auroc:.4f}  |  Pixel AUROC: {pxl_auroc:.4f}")

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def print_table(results: list[dict], vary: str) -> None:
    print(f"\n{'='*70}")
    print(f"Results — varying {vary}")
    print(f"{'='*70}")
    header = f"  {vary:>16s} | {'Img AUROC':>9s} | {'Pxl AUROC':>9s} | {'Mem Bank':>8s} | {'Train(s)':>8s}"
    print(header)
    print("  " + "-" * 66)
    for r in results:
        val       = r[vary]
        img_auroc = r["image_auroc"]
        pxl_auroc = r.get("pixel_auroc") or float("nan")
        mem       = r["memory_bank_size"]
        t         = r["train_time_s"]
        print(f"  {val:>16} | {img_auroc:9.4f} | {pxl_auroc:9.4f} | {mem:>8,d} | {t:>8.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PatchCore hyperparameter ablation study")
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--category", type=str, default=None,
                        help="MVTec AD category to ablate on (default: from config)")
    parser.add_argument("--skip-knn",     action="store_true", help="Skip num_neighbors ablation")
    parser.add_argument("--skip-coreset", action="store_true", help="Skip coreset_ratio ablation")
    args = parser.parse_args()

    config   = load_config(args.config)
    category = args.category or config["dataset"]["category"]

    print(f"\nPatchCore Ablation Study — category: {category}")
    print(f"Backbone: {config['model']['backbone']}  |  Device: {config['device']}")

    all_results = []

    if not args.skip_coreset:
        coreset_results = run_coreset_ablation(config, category)
        all_results.extend(coreset_results)
        print_table(coreset_results, "coreset_ratio")

    if not args.skip_knn:
        knn_results = run_knn_ablation(config, category)
        all_results.extend(knn_results)
        print_table(knn_results, "num_neighbors")

    # Save results
    out_dir = Path(config["output"]["dir"]) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
