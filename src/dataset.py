"""
Dataset loader for PCOSGen-style data: a flat folder of images plus a
separate CSV/XLSX file mapping filename -> class label.

Expected label file columns (PCOSGen's actual column names):
    imagePath
    Class label (whether the ultrasound images appears normal or has some abnormality)
    Class label (whether polycsytic ovary is visible or not visible)

We use the first label column as the binary target:
    "Appears normal"   -> 0
    "Appears abnormal" -> 1
Rows with any other value (e.g. "Unknown") or a missing imagePath are dropped.

Usage:
    ds = PCOSDataset(
        image_dir="data/pcosgen_train/images",
        label_file="data/pcosgen_train/class_label.xlsx",
        transform=TRAIN_TRANSFORM,
    )
"""
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
NORMAL_COL = "Class label (whether the ultrasound images appears normal or has some abnormality)"
VALID_LABEL_MAP = {
    "appears normal": 0,
    "appears abnormal": 1,
}

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_label_file(label_file: str) -> pd.DataFrame:
    label_file = Path(label_file)
    if label_file.suffix.lower() == ".csv":
        df = pd.read_csv(label_file)
    elif label_file.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(label_file)
    else:
        raise ValueError(f"Unsupported label file type: {label_file.suffix}")
    return df


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not a usable image"
        return False


class PCOSDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_file: str,
        transform=None,
        exclude_paths: Optional[Set[str]] = None,
        validate: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        exclude_paths = exclude_paths or set()

        df = _load_label_file(label_file)

        # Drop rows with missing imagePath
        df = df.dropna(subset=["imagePath"])

        # Normalize label text: strip whitespace, lowercase, map to 0/1
        df["_label_clean"] = df[NORMAL_COL].astype(str).str.strip().str.lower()
        df = df[df["_label_clean"].isin(VALID_LABEL_MAP.keys())]
        df["_label"] = df["_label_clean"].map(VALID_LABEL_MAP)

        self.samples = []
        n_missing_file = 0
        n_corrupt = 0
        for _, row in df.iterrows():
            img_path = self.image_dir / str(row["imagePath"]).strip()
            if not img_path.exists():
                n_missing_file += 1
                continue
            if str(img_path) in exclude_paths:
                continue
            if validate and not _is_readable_image(img_path):
                n_corrupt += 1
                continue
            self.samples.append((img_path, int(row["_label"])))

        if n_missing_file:
            print(
                f"[PCOSDataset] Warning: {n_missing_file} filenames in "
                f"{label_file} were not found in {self.image_dir} and were skipped."
            )
        if n_corrupt:
            print(f"[PCOSDataset] Warning: {n_corrupt} images could not be opened (corrupt/invalid) and were skipped.")

        if not self.samples:
            raise RuntimeError(
                f"No valid (image, label) pairs found. Checked {len(df)} label "
                f"rows against images in {self.image_dir}."
            )

        n_pos = sum(l for _, l in self.samples)
        print(
            f"[PCOSDataset] Loaded {len(self.samples)} images from {self.image_dir} "
            f"({len(self.samples) - n_pos} normal / {n_pos} abnormal)."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)


class FolderPCOSDataset(Dataset):
    """For datasets organized as class subfolders instead of a label file,
    e.g. many Kaggle PCOS sets:
        data_dir/
            infected/*.jpg      (or "abnormal", "PCOS", etc.)
            notinfected/*.jpg   (or "normal", "healthy", etc.)

    Pass the two folder names explicitly since they vary by dataset --
    positive_class_dir maps to label 1, negative_class_dir maps to label 0.
    """

    def __init__(
        self,
        data_dir: str,
        positive_class_dir: str,
        negative_class_dir: str,
        transform=None,
        exclude_paths: Optional[Set[str]] = None,
        validate: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        exclude_paths = exclude_paths or set()
        self.samples = []
        n_corrupt = 0

        for label, folder_name in [(0, negative_class_dir), (1, positive_class_dir)]:
            class_dir = self.data_dir / folder_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected folder not found: {class_dir}")
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"} or str(img_path) in exclude_paths:
                    continue
                if validate and not _is_readable_image(img_path):
                    n_corrupt += 1
                    continue
                self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.data_dir}.")
        if n_corrupt:
            print(f"[FolderPCOSDataset] Warning: {n_corrupt} images could not be opened (corrupt/invalid) and were skipped.")

        n_pos = sum(l for _, l in self.samples)
        print(
            f"[FolderPCOSDataset] Loaded {len(self.samples)} images from {self.data_dir} "
            f"({len(self.samples) - n_pos} negative / {n_pos} positive)."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)
