"""MVTec AD PyTorch Dataset."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import get_transforms

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


class MVTecADDataset(Dataset):
    """MVTec Anomaly Detection dataset.

    Args:
        root: Root directory containing the extracted MVTec AD data.
        category: Product category (e.g. 'bottle').
        split: 'train' or 'test'.
        image_size: Size to resize images to.
        center_crop: Size of the center crop.
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        image_size: int = 224,
        center_crop: int = 224,
    ):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Choose from: {CATEGORIES}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be 'train', 'val', or 'test', got '{split}'")

        self.root = Path(root) / category / split
        self.category = category
        self.split = split

        transforms = get_transforms(image_size, center_crop)
        self.image_transform = transforms["image"]
        self.mask_transform = transforms["mask"]

        self.image_paths, self.labels, self.mask_paths = self._load_samples()

    def _load_samples(self) -> tuple[list[Path], list[int], list[Path | None]]:
        """Scan directory structure to build sample list."""
        image_paths = []
        labels = []
        mask_paths = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}. "
                f"Run 'python scripts/download_dataset.py' first."
            )

        for defect_dir in sorted(self.root.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_normal = defect_type == "good"

            for img_path in sorted(defect_dir.glob("*.png")):
                image_paths.append(img_path)
                labels.append(0 if is_normal else 1)

                if is_normal or self.split == "train":
                    mask_paths.append(None)
                else:
                    # Ground truth masks are in ground_truth/<defect_type>/
                    mask_name = img_path.stem + "_mask.png"
                    mask_path = (
                        self.root.parent / "ground_truth" / defect_type / mask_name
                    )
                    mask_paths.append(mask_path if mask_path.exists() else None)

        return image_paths, labels, mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)

        label = self.labels[idx]

        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "image_path": str(self.image_paths[idx]),
        }
