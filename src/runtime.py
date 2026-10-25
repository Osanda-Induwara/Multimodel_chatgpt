"""Runtime setup for local Python and Google Colab."""
from __future__ import annotations

from pathlib import Path
import os


COLAB_DATA_ROOT = Path("/content/drive/MyDrive/TeaProject/dataset/Raw Dataset")
KAGGLE_DATASET_ROOTS = [
    Path("/kaggle/input/tea-liquor-dataset"),
    Path("/kaggle/input/tea_liquor_dataset"),
]
LOCAL_DATA_ROOT = Path(r"C:\Users\ASUS\Desktop\Teacc\dataset\Raw Dataset")


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore # Available only in Colab runtimes.
        return True
    except ImportError:
        return False


def running_in_kaggle() -> bool:
    """Return True only inside a Kaggle notebook runtime."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def find_raw_dataset(dataset_root: Path) -> Path:
    """Support either a direct Raw Dataset upload or an enclosing dataset folder."""
    candidates = [dataset_root / "Raw Dataset", dataset_root / "dataset" / "Raw Dataset"]
    candidates.extend(path for path in dataset_root.glob("**/Raw Dataset") if path.is_dir())
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Raw Dataset was not found inside {dataset_root}. "
        "Attach the Kaggle dataset 'tea_liquor_dataset' as notebook input."
    )


def get_data_root() -> Path:
    """Return the Raw Dataset folder for Kaggle, Colab, or local development."""
    if running_in_kaggle():
        for dataset_root in KAGGLE_DATASET_ROOTS:
            if dataset_root.is_dir():
                return find_raw_dataset(dataset_root)
        # Kaggle can alter a private dataset's mount slug. Fall back to inspecting
        # every attached input and selecting the one that actually contains images.
        input_root = Path("/kaggle/input")
        for dataset_root in input_root.iterdir():
            if dataset_root.is_dir():
                try:
                    return find_raw_dataset(dataset_root)
                except FileNotFoundError:
                    continue
        raise FileNotFoundError(
            "No attached Kaggle input contains a 'Raw Dataset' folder. "
            "Use Add Input and attach tea_liquor_dataset."
        )
    if running_in_colab():
        from google.colab import drive  # type: ignore
        # This is a no-op when Drive is already mounted; otherwise Colab prompts once.
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        if not COLAB_DATA_ROOT.exists():
            raise FileNotFoundError(
                "Raw Dataset not found. Expected: "
                "/content/drive/MyDrive/TeaProject/dataset/Raw Dataset"
            )
        return COLAB_DATA_ROOT
    if not LOCAL_DATA_ROOT.exists():
        raise FileNotFoundError(f"Raw Dataset not found: {LOCAL_DATA_ROOT}")
    return LOCAL_DATA_ROOT


def get_cache_root() -> Path:
    """Return fast ephemeral storage for preprocessed runtime images."""
    if running_in_kaggle():
        return Path("/kaggle/working/tea_liquor_cache")
    if running_in_colab():
        return Path("/content/tea_liquor_cache")
    return Path(__file__).resolve().parents[1] / "data" / "tea_liquor_cache"
