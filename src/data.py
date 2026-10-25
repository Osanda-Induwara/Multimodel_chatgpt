"""Dataset discovery, leakage-safe splitting, and tea-circle preprocessing."""
from __future__ import annotations

import re
import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from .config import GRADE_LABELS, MANIFEST_PATH, REGION_LABELS, SEED
from .runtime import get_cache_root, get_data_root

FOLDER_PATTERN = re.compile(r"^(?P<region>[A-Z]+)_(?P<grade>[A-Z0-9]+)_(?P<sample>\d+)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_manifest(data_root: Path) -> pd.DataFrame:
    """Read labels from sample-folder names; retain only requested grades/regions."""
    records = []
    for image_path in sorted(data_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = FOLDER_PATTERN.match(image_path.parent.name.upper())
        if not match:
            continue
        values = match.groupdict()
        if values["grade"] not in GRADE_LABELS or values["region"] not in REGION_LABELS:
            continue
        # Camera is the dataset's top-level directory; group only physical sample.
        camera = image_path.relative_to(data_root).parts[0]
        group_id = "{region}_{grade}_{sample}".format(**values)
        records.append({
            "path": str(image_path.resolve()), "region": values["region"],
            "grade": values["grade"], "sample_id": group_id, "camera": camera,
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No requested images found. Check --data-root and folder naming.")
    return frame.sort_values("path").reset_index(drop=True)


def load_or_build_manifest(data_root: Path | None = None) -> pd.DataFrame:
    """Load the manifest, creating it automatically when first run in Colab/local."""
    if MANIFEST_PATH.exists():
        frame = pd.read_csv(MANIFEST_PATH)
        # A manifest copied from Windows contains Windows image paths, so rebuild it
        # in Colab rather than failing later inside the training DataLoader.
        if not frame.empty and Path(frame.iloc[0].path).exists():
            return frame
    data_root = data_root or get_data_root()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Raw Dataset not found at {data_root}. In Colab, mount Google Drive and "
            "place it at /content/drive/MyDrive/TeaProject/dataset/Raw Dataset."
        )
    frame = build_manifest(data_root)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST_PATH, index=False)
    print(f"Created manifest with {len(frame)} images: {MANIFEST_PATH}")
    return frame


def grouped_split(frame: pd.DataFrame, validation_fraction: float = 0.2, seed: int = SEED):
    """Split at physical-sample level, preventing camera-pair leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, valid_idx = next(splitter.split(frame, groups=frame["sample_id"]))
    return frame.iloc[train_idx].copy(), frame.iloc[valid_idx].copy()


def crop_tea_liquor(image_bgr: np.ndarray, output_size: int = 224) -> np.ndarray:
    """Find the bowl circle, remove bright internal glare, and return RGB circular crop."""
    height, width = image_bgr.shape[:2]
    # Detect on a small preview: Hough circles on phone-resolution originals is slow.
    scale = min(1.0, 800 / max(height, width))
    preview = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_h, small_w = preview.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(small_h, small_w) // 2,
                               param1=100, param2=35, minRadius=int(min(small_h, small_w) * .20),
                               maxRadius=int(min(small_h, small_w) * .49))
    if circles is None:
        cx, cy, radius = width // 2, height // 2, int(min(height, width) * .38)
    else:
        candidates = np.round(circles[0] / scale).astype(int)
        # Prefer circle nearest the image centre: it is the expected bowl position.
        cx, cy, radius = min(candidates, key=lambda c: (c[0] - width / 2) ** 2 + (c[1] - height / 2) ** 2)
    radius = max(8, min(radius, cx, cy, width - cx - 1, height - cy - 1))
    y0, y1, x0, x1 = cy - radius, cy + radius, cx - radius, cx + radius
    crop = image_bgr[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.circle(mask, (radius, radius), int(radius * .96), 255, -1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Restrict bright-glare repair to low-saturation pixels within tea. A camera can
    # also appear as a darker central patch, detected against a local illumination map.
    bright_glare = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 175) & (mask > 0)).astype(np.uint8) * 255
    local_value = cv2.GaussianBlur(hsv[:, :, 2], (0, 0), max(5, radius // 5))
    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    central = (xx - radius) ** 2 + (yy - radius) ** 2 < (radius * .43) ** 2
    dark_reflection = ((local_value.astype(np.int16) - hsv[:, :, 2].astype(np.int16) > 12) & central & (mask > 0)).astype(np.uint8) * 255
    dark_reflection = cv2.morphologyEx(dark_reflection, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    reflection_mask = cv2.bitwise_or(bright_glare, dark_reflection)
    if np.count_nonzero(reflection_mask) > 0:
        crop = cv2.inpaint(crop, reflection_mask, 3, cv2.INPAINT_TELEA)
    crop[mask == 0] = 0
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def read_for_preprocessing(path: str | Path, max_dimension: int = 1000) -> np.ndarray:
    """Efficiently decode a downsampled JPEG; 12,000-pixel originals are unnecessary for 224px training."""
    # The images are trusted research files; this suppresses only Pillow's warning
    # for their known high-resolution source photographs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            source.draft("RGB", (max_dimension, max_dimension))
            source = source.convert("RGB")
            source.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)


class TeaDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, task: str, transform=None, image_size: int = 224, cache_dir: Path | None = None):
        self.frame = frame.reset_index(drop=True)
        self.task = task
        self.transform = transform
        self.image_size = image_size
        self.labels = GRADE_LABELS if task == "grade" else REGION_LABELS
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        self.cache_dir = cache_dir or get_cache_root()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        source_path = Path(row.path)
        cache_key = hashlib.sha1(f"{source_path}|{source_path.stat().st_mtime_ns}|{self.image_size}".encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.jpg"
        if cache_path.exists():
            with Image.open(cache_path) as cached:
                image = cached.convert("RGB")
        else:
            crop = crop_tea_liquor(read_for_preprocessing(source_path), self.image_size)
            image = Image.fromarray(crop)
            image.save(cache_path, "JPEG", quality=95)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.label_to_index[row[self.task]], dtype=torch.long), row.path
