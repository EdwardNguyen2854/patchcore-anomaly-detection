"""Auto-download and extract the MVTec AD dataset."""

import hashlib
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# The official MVTec AD download link is no longer available.
# Download from Kaggle instead: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
# Or use: make extract SOURCE=path/to/mvtec_anomaly_detection.tar.xz
MVTEC_AD_URL = None  # Deprecated - download from Kaggle
EXPECTED_MD5 = None  # MD5 check skipped; verify file size instead
EXPECTED_SIZE_MB = 4700  # ~4.7 GB


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def verify_file(filepath: Path) -> bool:
    """Verify downloaded file by checking minimum size."""
    size_mb = filepath.stat().st_size / (1024 * 1024)
    if size_mb < EXPECTED_SIZE_MB * 0.9:
        print(f"Warning: File size ({size_mb:.0f} MB) is smaller than expected ({EXPECTED_SIZE_MB} MB)")
        return False
    return True


def extract_tar(filepath: Path, dest: Path) -> None:
    """Extract a tar.xz archive."""
    print(f"Extracting {filepath.name} to {dest}...")
    with tarfile.open(filepath, "r:xz") as tar:
        tar.extractall(dest)
    print("Extraction complete.")


def download_mvtec_ad(root: str = "data", force: bool = False) -> Path:
    """Download and extract the MVTec AD dataset.

    Note: The official MVTec download link is no longer available.
    Please download from Kaggle: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
    Then use: python scripts/download_dataset.py --root data --source /path/to/file.tar.xz

    Args:
        root: Root directory to store the dataset.
        force: Re-download even if already present.

    Returns:
        Path to the extracted dataset directory.
    """
    if MVTEC_AD_URL is None:
        raise RuntimeError(
            "The official MVTec AD download link is no longer available.\n"
            "Please download the dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/ipythonx/mvtec-ad\n"
            "Then extract using:\n"
            "  python scripts/download_dataset.py --root data --source /path/to/mvtec_anomaly_detection.tar.xz"
        )

    root = Path(root)
    dataset_dir = root / "mvtec_anomaly_detection"
    archive_path = root / "mvtec_anomaly_detection.tar.xz"

    # Check if already extracted
    if dataset_dir.exists() and not force:
        num_categories = sum(1 for d in dataset_dir.iterdir() if d.is_dir())
        if num_categories >= 15:
            print(f"Dataset already exists at {dataset_dir} ({num_categories} categories found).")
            return dataset_dir

    root.mkdir(parents=True, exist_ok=True)

    # Download
    if not archive_path.exists() or force:
        print(f"Downloading MVTec AD dataset to {archive_path}...")
        download_file(MVTEC_AD_URL, archive_path)
        if not verify_file(archive_path):
            print("Download may be incomplete. Re-run with --force to retry.")
    else:
        print(f"Archive already exists at {archive_path}.")

    # Extract
    extract_tar(archive_path, root)

    if not dataset_dir.exists():
        raise RuntimeError(
            f"Expected dataset directory {dataset_dir} not found after extraction."
        )

    print(f"MVTec AD dataset ready at {dataset_dir}")
    return dataset_dir
