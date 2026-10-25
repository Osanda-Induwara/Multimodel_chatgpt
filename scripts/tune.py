"""Small, dependency-free hyperparameter search; compare its output with a baseline run."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MANIFEST_PATH
from src.training import run_training

parser = argparse.ArgumentParser(description="Run a small learning-rate/batch-size search.")
parser.add_argument("--task", choices=["grade", "region"], required=True)
parser.add_argument("--model", default="efficientnet", choices=["efficientnet", "mobilenet", "resnet", "shufflenet", "squeezenet"])
parser.add_argument("--trials", type=int, default=6, help="Number of presets to evaluate.")
parser.add_argument("--epochs", type=int, default=10, help="Short search-run epoch count.")
args = parser.parse_args()
if not MANIFEST_PATH.exists(): raise FileNotFoundError("Run scripts/build_manifest.py first.")

presets = [
    (1e-3, 0.0, 16), (3e-4, 1e-4, 16), (1e-4, 1e-4, 16),
    (3e-4, 1e-5, 8), (1e-4, 1e-5, 8), (5e-4, 1e-4, 32),
]
frame = pd.read_csv(MANIFEST_PATH)
results = []
for index, (lr, wd, batch_size) in enumerate(presets[:args.trials], start=1):
    run_dir, config = run_training(frame, args.task, args.model, args.epochs, batch_size, lr, wd, tag=f"tuned_trial_{index}")
    results.append({**config, "run_dir": str(run_dir)})
    print(f"trial {index}: validation accuracy {config['validation_accuracy']:.3f}")
summary = pd.DataFrame(results).sort_values("validation_accuracy", ascending=False)
summary.to_csv(Path("results") / args.task / f"tuning_summary_{args.model}.csv", index=False)
print("\nBest setting:\n", summary.iloc[0][["learning_rate", "weight_decay", "batch_size", "validation_accuracy", "run_dir"]])
