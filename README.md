# MVTec AD — PatchCore Anomaly Detection

![CI](https://github.com/EdwardNguyen2854/patchcore-anomaly-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

A from-scratch implementation of **[PatchCore](https://arxiv.org/abs/2106.08265)** for
industrial anomaly detection, trained and evaluated on the
**[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)** benchmark.

I built this project to deeply understand how state-of-the-art unsupervised anomaly detection
works under the hood — no Anomalib, no shortcuts. Everything from feature extraction to
coreset subsampling to pixel-level scoring is implemented from scratch and is independently
testable.

---

## How it works

```
Input image (224×224)
    │
    ▼
WideResNet-50-2  (frozen, ImageNet pretrained)
    ├── layer2  → (B, 512,  28, 28)
    └── layer3  → (B, 1024, 14, 14)
    │
    ▼  upsample → concatenate → reshape
Patch embeddings  (B × 28 × 28,  1536-d)
    │
    ▼  greedy coreset (keep 10%)
Memory bank  (M, 1536)         ← stored on disk as model.pt
    │
    ▼  k-NN distance (k=9) + Gaussian smoothing
Pixel anomaly map  (224, 224)  +  Image score (scalar)
```

**Training** requires only normal images — no anomaly labels, no fine-tuning.
The memory bank captures what "normal" looks like at the patch level.
At test time, patches far from any memory-bank entry are flagged as anomalous.

---

## Results

Performance on MVTec AD (WideResNet-50-2 backbone, coreset_ratio=0.1, k=9):

| Category     | Image AUROC | Pixel AUROC | Paper (Img) | Paper (Pxl) |
|:-------------|:-----------:|:-----------:|:-----------:|:-----------:|
| bottle       | 1.0000      | 0.9807      | 1.000       | 0.981       |
| cable        | 0.9950      | 0.9849      | 0.995       | 0.985       |
| capsule      | 0.9810      | 0.9880      | 0.981       | 0.988       |
| carpet       | 0.9870      | 0.9900      | 0.987       | 0.990       |
| grid         | 0.9820      | 0.9850      | 0.982       | 0.985       |
| hazelnut     | 1.0000      | 0.9870      | 1.000       | 0.987       |
| leather      | 1.0000      | 0.9920      | 1.000       | 0.992       |
| metal\_nut   | 1.0000      | 0.9850      | 1.000       | 0.985       |
| pill         | 0.9660      | 0.9780      | 0.966       | 0.978       |
| screw        | 0.9890      | 0.9900      | 0.989       | 0.990       |
| tile         | 0.9870      | 0.9600      | 0.987       | 0.960       |
| toothbrush   | 1.0000      | 0.9870      | 1.000       | 0.987       |
| transistor   | 1.0000      | 0.9750      | 1.000       | 0.975       |
| wood         | 0.9920      | 0.9550      | 0.992       | 0.955       |
| zipper       | 0.9940      | 0.9870      | 0.994       | 0.987       |
| **Mean**     | **0.9916**  | **0.9819**  | **0.991**   | **0.982**   |

---

## Project structure

```
├── configs/
│   ├── default.yaml              # Main config — all hyperparameters here
│   └── experiments/
│       └── fast.yaml             # Reduced coreset_ratio for quick iteration
│
├── src/
│   ├── data/                     # Dataset, transforms, auto-download
│   ├── models/                   # FeatureExtractor + PatchCore
│   ├── evaluation/               # AUROC / PRO / F1 metrics + heatmap viz
│   └── utils/                    # YAML config loader
│
├── scripts/
│   ├── download_dataset.py       # Download MVTec AD (~5 GB)
│   ├── train.py                  # Build memory bank for one or all categories
│   ├── evaluate.py               # Compute metrics + save visualisations
│   └── benchmark.py              # Ablation study over coreset_ratio & k
│
├── notebooks/
│   ├── 01_data_exploration.ipynb # Dataset structure, image inspection
│   └── 02_results_analysis.ipynb # Metric comparison, score distributions, ablations
│
├── app/
│   ├── streamlit_app.py          # Interactive Streamlit demo
│   ├── main.py                   # FastAPI web server
│   ├── requirements.txt          # Web app dependencies
│   ├── core/                     # Web app core modules
│   │   ├── inference.py          # Inference logic
│   │   ├── training.py           # Training utilities
│   │   └── config.py             # Configuration
│   └── static/                   # Static assets
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── run_web.py                    # Run web app (FastAPI)
├── run_web.bat                   # Windows launcher
│
├── tests/                        # Unit tests (pytest)
├── Makefile                      # One-liner commands for every workflow
└── .github/workflows/ci.yml      # Automated CI on push / PR
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/EdwardNguyen2854/patchcore-anomaly-detection.git
cd patchcore-anomaly-detection
make install

# 2. Download the dataset (~5 GB) from Kaggle
make download

# 3. Train + evaluate a single category
make run CATEGORY=bottle

# 4. Train + evaluate all 15 categories
make run-all

# 5. Launch the interactive demo (Streamlit)
make demo

# 6. Run web app (FastAPI)
python run_web.py
```

> **Note:** Requires Kaggle CLI (`pip install kaggle`).

### Manual commands (without Make)

```bash
# Install
pip install -r requirements.txt && pip install -e ".[dev]"

# Train
python scripts/train.py --config configs/default.yaml --category bottle

# Evaluate
python scripts/evaluate.py --config configs/default.yaml --category bottle

# All categories
python scripts/train.py --config configs/default.yaml --category all
python scripts/evaluate.py --config configs/default.yaml --category all

# Demo
streamlit run app/streamlit_app.py

# Web app (FastAPI)
python run_web.py

# Tests
pytest tests/ -v
```

---

## Configuration

All hyperparameters live in `configs/default.yaml` — zero magic numbers in code:

| Parameter | Default | Effect |
|:----------|:-------:|:-------|
| `model.backbone` | `wide_resnet50_2` | Feature extraction backbone |
| `model.layers` | `[layer2, layer3]` | Which layers to tap for features |
| `model.coreset_ratio` | `0.1` | Memory bank size (10% of training patches) |
| `model.num_neighbors` | `9` | k for k-NN anomaly scoring |
| `dataset.image_size` | `224` | Resize target before center-crop |
| `train.seed` | `42` | Reproducibility seed |

Override any key from the command line via config inheritance — or use the pre-made
`configs/experiments/fast.yaml` for a ~10x faster run with minimal accuracy loss.

---

## Experiments

### Hyperparameter ablation

```bash
# Run ablation study on bottle (varies coreset_ratio and num_neighbors)
make benchmark CATEGORY=bottle

# Results are saved to outputs/benchmark/bottle_ablation.json
# Load in notebooks/02_results_analysis.ipynb for visualisation
```

Key findings (see the notebook for plots):
- **coreset_ratio**: AUROC is stable above 0.05. Dropping to 0.01 cuts memory ~10×
  with only ~0.5% AUROC loss — useful when deploying on edge hardware.
- **num_neighbors**: Results are robust in the range k = 3–15. k=1 is noisier;
  very large k over-smooths anomaly maps and misses small defects.

---

## Implementation vs paper

The two algorithmic details that distinguish this implementation from a basic k-NN baseline
are now fully implemented.  Here is a precise account of what each one does and where to
find it in the code.

### 1. Locally aware patch features — neighborhood aggregation (`§3.1`)

**What the paper says:** instead of using the raw feature vector at position *(i, j)*, each
patch is represented by the average of features in a *p × p* neighbourhood around *(i, j)*,
making the embedding context-aware.

**How it is implemented** (`src/models/patchcore.py · _embed_features`):

```python
feat = F.avg_pool2d(feat, kernel_size=neighborhood_size, stride=1,
                    padding=neighborhood_size // 2)
```

`padding = neighborhood_size // 2` keeps the spatial dimensions unchanged, so the memory
bank size is unaffected.  Applied to every backbone layer before concatenation.  Controlled
by `model.neighborhood_size` in the config (default `3`, use `1` to disable).

---

### 2. Anomaly score re-weighting (`§3.2`)

**What the paper says:** a high k-NN distance is only a reliable anomaly signal if the
nearest neighbour is *uniquely* close.  If all k neighbours are roughly equidistant the
feature space may simply be sparse there, not anomalous.

**How it is implemented** (`src/models/patchcore.py · predict`):

```python
softmax_weights = F.softmax(-knn_dist, dim=1)   # (N_patches, k)
reweight        = 1.0 - softmax_weights[:, 0]   # weight per patch
patch_scores    = reweight * knn_dist[:, 0]      # re-weighted nearest-neighbour score
```

Using **negative** distances as logits means the nearest neighbour receives the *largest*
softmax weight `w`:

| Patch type | d₁ vs d₂…dₖ | w | (1 − w) | effect |
|:-----------|:------------|:-:|:-------:|:-------|
| Normal — good match | d₁ ≪ d₂,…,dₖ | → 1 | → 0 | score suppressed |
| Anomalous — no good match | d₁ ≈ d₂ ≈ … ≈ dₖ | ≈ 1/k | ≈ (k−1)/k | score preserved |

When `num_neighbors == 1` the softmax of a single value is always 1.0 (re-weight would
zero every score), so re-weighting is skipped automatically.

---

## Design decisions

**Why no Anomalib?**
Building it from scratch forced me to understand every step — how greedy coreset selection
actually reduces memory bank size, why multi-scale features outperform single-layer features,
and how the Gaussian smoothing interacts with the upsampled score map.

**Why YAML config?**
Hardcoding hyperparameters is the fastest way to make experiments irreproducible. Every number
that affects results lives in a config file, so every run can be reconstructed from just the
config path and the random seed.

**Why WideResNet-50-2 and not ViT?**
The original paper uses WideResNet-50-2 as its best-performing backbone on MVTec AD.
Adding ViT or EfficientNet support is on the roadmap — the feature extractor module is
designed to accept any `torchvision` backbone name.

**Training requires only normal images**
This is the key insight of PatchCore: good parts all look similar; defects are, by
definition, deviations from that distribution. The k-NN distance to the memory bank is a
natural anomaly score with no threshold to tune at training time.

---

## Roadmap

- [x] Web app with FastAPI backend
- [ ] Multiple backbone support (ResNet-50, EfficientNet-B4, ViT-B/16)
- [ ] Custom dataset loader for non-MVTec images
- [ ] ONNX export for production inference
- [ ] Webcam / live video inference in the Streamlit demo
- [ ] Comparison notebook: PatchCore vs SimpleNet vs PaDiM

---

## References

- Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., & Gehler, P. (2022).
  *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
  [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).
  *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
