import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_ROOT
from src.data import crop_tea_liquor, load_or_build_manifest, read_for_preprocessing

parser = argparse.ArgumentParser(description="Create a visual quality check for bowl-circle preprocessing.")
parser.add_argument("--limit", type=int, default=24)
args = parser.parse_args()
rows = load_or_build_manifest().groupby("sample_id", group_keys=False).head(1).head(args.limit)
columns = 6; figure, axes = plt.subplots((len(rows) + columns - 1) // columns, columns, figsize=(15, 3 * ((len(rows) + columns - 1) // columns)), squeeze=False)
for axis, (_, row) in zip(axes.flat, rows.iterrows()):
    image = read_for_preprocessing(row.path); axis.imshow(crop_tea_liquor(image)); axis.set_title(f"{row.region} {row.grade}\n{row.sample_id}", fontsize=8); axis.axis("off")
for axis in list(axes.flat)[len(rows):]: axis.axis("off")
output = RESULTS_ROOT / "preprocessing_preview.png"; output.parent.mkdir(exist_ok=True)
figure.tight_layout(); figure.savefig(output, dpi=160); print(f"Saved {output}")
