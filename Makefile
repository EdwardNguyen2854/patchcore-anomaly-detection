.PHONY: install test download train train-all evaluate evaluate-all benchmark demo clean help

PYTHON  := python
CONFIG  := configs/default.yaml
CATEGORY := bottle

# ─── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# ─── Quality ──────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

# ─── Data ─────────────────────────────────────────────────────────────────────

download:
	$(PYTHON) scripts/download_dataset.py

# ─── Training & Evaluation ────────────────────────────────────────────────────

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --category $(CATEGORY)

train-all:
	$(PYTHON) scripts/train.py --config $(CONFIG) --category all

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --category $(CATEGORY)

evaluate-all:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --category all

# Full pipeline for a single category
run: train evaluate

# Full pipeline for all categories
run-all: train-all evaluate-all

# ─── Experiments ──────────────────────────────────────────────────────────────

benchmark:
	$(PYTHON) scripts/benchmark.py --config $(CONFIG) --category $(CATEGORY)

# ─── Demo ─────────────────────────────────────────────────────────────────────

demo:
	streamlit run app/streamlit_app.py

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache

clean-outputs:
	rm -rf outputs/

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "MVTec AD Anomaly Detection — Available commands"
	@echo "================================================"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install all dependencies"
	@echo ""
	@echo "  Data"
	@echo "    make download         Download MVTec AD from Kaggle (~5 GB)"
	@echo "    make extract         Extract from local archive (set SOURCE=path/to/file)"
	@echo ""
	@echo "  Training & Evaluation"
	@echo "    make train            Train on CATEGORY (default: bottle)"
	@echo "    make train-all        Train on all 15 categories"
	@echo "    make evaluate         Evaluate on CATEGORY"
	@echo "    make evaluate-all     Evaluate on all 15 categories"
	@echo "    make run              Train + evaluate one category"
	@echo "    make run-all          Train + evaluate all categories"
	@echo ""
	@echo "  Experiments"
	@echo "    make benchmark        Ablation study on CATEGORY"
	@echo ""
	@echo "  Demo"
	@echo "    make demo             Launch Streamlit web demo"
	@echo ""
	@echo "  Utilities"
	@echo "    make test             Run unit tests"
	@echo "    make clean            Remove cache files"
	@echo "    make clean-outputs    Remove all trained models and results"
	@echo ""
	@echo "  Override category:  make train CATEGORY=cable"
	@echo "  Override config:    make train CONFIG=configs/experiments/fast.yaml"
	@echo ""
