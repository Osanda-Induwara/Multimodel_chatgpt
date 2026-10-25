"""Runtime setup for local Python and Google Colab."""
from __future__ import annotations

from pathlib import Path


COLAB_DATA_ROOT = Path("/content/drive/MyDrive/TeaProject/dataset/Raw Dataset")
LOCAL_DATA_ROOT = Path(r"C:\Users\ASUS\Desktop\Teacc\dataset\Raw Dataset")


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore # Available only in Colab runtimes.
        return True
    except ImportError:
        return False


def get_data_root() -> Path:
    """Mount Drive on Colab and return the project's Raw Dataset folder."""
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
    """Return fast ephemeral storage for preprocessed Colab images."""
    if running_in_colab():
        return Path("/content/tea_liquor_cache")
    return Path(__file__).resolve().parents[1] / "data" / "tea_liquor_cache"
