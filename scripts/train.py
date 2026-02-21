"""Training entrypoint for PatchCore anomaly detection."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CATEGORIES, MVTecADDataset
from src.models.patchcore import PatchCore
from src.utils.config import load_config


def train_category(config: dict, category: str) -> dict:
    """Train PatchCore on a single category.

    Args:
        config: Configuration dict.
        category: MVTec AD category name.

    Returns:
        Dict with training summary.
    """
    print(f"\n{'='*60}")
    print(f"Training PatchCore on category: {category}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load training data
    dataset = MVTecADDataset(
        root=config["dataset"]["root"],
        category=category,
        split="train",
        image_size=config["dataset"]["image_size"],
        center_crop=config["dataset"]["center_crop"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )
    print(f"Training samples: {len(dataset)}")

    # Initialize model
    model = PatchCore(
        backbone=config["model"]["backbone"],
        layers=config["model"]["layers"],
        pretrained=config["model"]["pretrained"],
        neighborhood_size=config["model"].get("neighborhood_size", 3),
        coreset_ratio=config["model"]["coreset_ratio"],
        num_neighbors=config["model"]["num_neighbors"],
        device=config["device"],
    )

    # Build memory bank
    model.fit(dataloader, seed=config["train"]["seed"])

    # Save model
    output_dir = Path(config["output"]["dir"]) / category
    model_path = output_dir / "model.pt"
    model.save(str(model_path))

    elapsed = time.time() - start_time
    summary = {
        "category": category,
        "num_train_samples": len(dataset),
        "memory_bank_size": model.memory_bank.shape[0],
        "training_time_seconds": round(elapsed, 1),
    }

    # Save training summary
    summary_path = output_dir / "train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete in {elapsed:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--category", type=str, default=None, help="Category to train on (default: from config)")
    args = parser.parse_args()

    config = load_config(args.config)

    category = args.category or config["dataset"]["category"]

    if category == "all":
        categories = CATEGORIES
    else:
        categories = [category]

    results = []
    for cat in categories:
        summary = train_category(config, cat)
        results.append(summary)

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['category']:15s} | samples: {r['num_train_samples']:4d} | "
              f"memory bank: {r['memory_bank_size']:6,d} | time: {r['training_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
