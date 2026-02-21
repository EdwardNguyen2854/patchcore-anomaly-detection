"""Interactive Streamlit demo for PatchCore anomaly detection."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Allow running from app/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CATEGORIES
from src.data.transforms import get_transforms
from src.evaluation.visualize import create_heatmap_overlay, denormalize
from src.models.patchcore import PatchCore
from src.utils.config import load_config


@st.cache_resource
def load_model(category: str) -> PatchCore:
    """Load a trained PatchCore model (cached)."""
    config = load_config("configs/default.yaml")
    # neighborhood_size and num_neighbors are restored from the saved state by model.load()
    model = PatchCore(
        backbone=config["model"]["backbone"],
        layers=config["model"]["layers"],
        pretrained=config["model"]["pretrained"],
        device=config["device"],
    )
    model_path = Path(config["output"]["dir"]) / category / "model.pt"
    model.load(str(model_path))
    return model


def get_available_categories(config: dict) -> list[str]:
    """Get list of categories that have trained models."""
    output_dir = Path(config["output"]["dir"])
    available = []
    for cat in CATEGORIES:
        if (output_dir / cat / "model.pt").exists():
            available.append(cat)
    return available


def get_test_images(config: dict, category: str) -> list[Path]:
    """Get list of test images for a category."""
    test_dir = Path(config["dataset"]["root"]) / category / "test"
    if not test_dir.exists():
        return []
    images = sorted(test_dir.rglob("*.png"))
    return images


def main():
    st.set_page_config(page_title="MVTec AD Anomaly Detection", layout="wide")
    st.title("PatchCore Anomaly Detection")
    st.markdown("Detect manufacturing defects using PatchCore on the MVTec AD benchmark.")

    # Load config
    config = load_config("configs/default.yaml")

    # Sidebar
    st.sidebar.header("Settings")

    available = get_available_categories(config)
    if not available:
        st.error(
            "No trained models found. Run `python scripts/train.py` first."
        )
        return

    category = st.sidebar.selectbox("Category", available)

    # Load model
    with st.spinner(f"Loading model for {category}..."):
        model = load_model(category)

    transforms = get_transforms(
        config["dataset"]["image_size"],
        config["dataset"]["center_crop"],
    )

    # Image source
    st.sidebar.header("Image Source")
    source = st.sidebar.radio("Choose image source", ["Upload Image", "Test Set"])

    image_pil = None

    if source == "Upload Image":
        uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded:
            image_pil = Image.open(uploaded).convert("RGB")
    else:
        test_images = get_test_images(config, category)
        if test_images:
            selected = st.sidebar.selectbox(
                "Select test image",
                test_images,
                format_func=lambda p: f"{p.parent.name}/{p.name}",
            )
            image_pil = Image.open(selected).convert("RGB")
        else:
            st.sidebar.warning("No test images found. Download the dataset first.")

    if image_pil is None:
        st.info("Select or upload an image to run anomaly detection.")
        return

    # Preprocess
    image_tensor = transforms["image"](image_pil).unsqueeze(0)

    # Predict
    with torch.no_grad():
        image_scores, score_maps = model.predict(image_tensor)

    score = image_scores[0].item()
    score_map = score_maps[0].cpu().numpy()
    display_image = denormalize(image_tensor[0])

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(display_image, use_container_width=True)

    with col2:
        st.subheader("Anomaly Heatmap")
        overlay = create_heatmap_overlay(display_image, score_map)
        st.image(overlay, use_container_width=True)

    # Score display
    st.markdown("---")
    score_col1, score_col2 = st.columns(2)

    with score_col1:
        st.metric("Anomaly Score", f"{score:.4f}")

    with score_col2:
        # Load threshold from eval metrics if available
        metrics_path = Path(config["output"]["dir"]) / category / "eval_metrics.json"
        threshold = None
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            threshold = metrics.get("image_threshold")

        if threshold is not None:
            status = "ANOMALOUS" if score > threshold else "NORMAL"
            color = "red" if score > threshold else "green"
            st.markdown(
                f"**Prediction:** :{color}[{status}] (threshold: {threshold:.4f})"
            )
        else:
            st.info("Run evaluation to determine the optimal threshold.")


if __name__ == "__main__":
    main()
