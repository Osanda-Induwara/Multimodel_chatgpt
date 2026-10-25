from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The active location is selected after Colab Drive has been mounted.
DEFAULT_DATA_ROOT = Path(r"C:\Users\ASUS\Desktop\Teacc\dataset\Raw Dataset")
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
RESULTS_ROOT = PROJECT_ROOT / "results"
GRADE_LABELS = ["BOP", "BOPF", "OP", "DUST"]
REGION_LABELS = ["DI", "KA", "NU", "RU", "SB", "UP", "UV"]
SEED = 42
