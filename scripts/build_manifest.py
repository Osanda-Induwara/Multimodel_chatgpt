import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MANIFEST_PATH
from src.data import build_manifest
from src.runtime import get_data_root

parser = argparse.ArgumentParser(description="Create labels manifest from tea sample folders.")
parser.add_argument("--data-root", type=Path, default=None,
                    help="Raw Dataset path; inferred automatically in Colab/local when omitted.")
args = parser.parse_args()
data_root = args.data_root or get_data_root()
frame = build_manifest(data_root)
MANIFEST_PATH.parent.mkdir(exist_ok=True)
frame.to_csv(MANIFEST_PATH, index=False)
print(f"Saved {len(frame)} images from {frame.sample_id.nunique()} physical samples to {MANIFEST_PATH}")
print("\nGrade counts:\n", frame.grade.value_counts().sort_index())
print("\nRegion counts:\n", frame.region.value_counts().sort_index())
