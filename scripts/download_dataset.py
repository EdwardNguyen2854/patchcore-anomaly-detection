"""CLI script to download and extract the MVTec AD dataset."""

import argparse
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path
import shutil

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import extract_tar


def download_from_kaggle(root: Path) -> None:
    """Download dataset using Kaggle CLI."""
    print("Downloading MVTec AD from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "ipythonx/mvtec-ad", "-p", str(root), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then run: kaggle datasets download ipythonx/mvtec-ad"
        )


def extract_kaggle_zip(root: Path) -> Path:
    """Extract Kaggle zip file."""
    zip_path = root / "mvtec_anomaly_detection.zip"
    
    if not zip_path.exists():
        for f in root.glob("*.zip"):
            zip_path = f
            break
    
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(root)
        
        extracted_dir = root / "mvtec_ad"
        if extracted_dir.exists():
            final_dir = root / "mvtec_anomaly_detection"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.move(str(extracted_dir), str(final_dir))
        
        zip_path.unlink()
    
    return root / "mvtec_anomaly_detection"


def main():
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument("--root", type=str, default="data", help="Root directory for dataset storage")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--source", type=str, default=None, help="Path to local archive file")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.source:
        source_path = Path(args.source)
        if source_path.suffix == ".xz" or source_path.suffix == ".tar":
            print(f"Extracting {source_path} to {root}...")
            extract_tar(source_path, root)
        elif source_path.suffix == ".zip":
            print(f"Extracting {source_path} to {root}...")
            with zipfile.ZipFile(source_path, 'r') as z:
                z.extractall(root)
    else:
        download_from_kaggle(root)
        dataset_dir = extract_kaggle_zip(root)
        print(f"MVTec AD dataset ready at {dataset_dir}")
        return

    dataset_dir = root / "mvtec_anomaly_detection"
    if dataset_dir.exists():
        print(f"MVTec AD dataset ready at {dataset_dir}")
    else:
        print(f"Warning: Expected dataset directory {dataset_dir} not found.")


if __name__ == "__main__":
    main()
