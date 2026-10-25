import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_ROOT

parser = argparse.ArgumentParser(description="Compare baseline training against tuned trials.")
parser.add_argument("--task", choices=["grade", "region"], required=True)
args = parser.parse_args()
configs = []
for path in (RESULTS_ROOT / args.task).glob("*/config.json"):
    config = json.loads(path.read_text(encoding="utf-8"))
    config["run"] = path.parent.name
    configs.append(config)
if not configs:
    raise FileNotFoundError("No training results found for this task.")
table = pd.DataFrame(configs).sort_values("validation_accuracy", ascending=False)
output = RESULTS_ROOT / args.task
table.to_csv(output / "experiment_comparison.csv", index=False)
fig, axis = plt.subplots(figsize=(max(7, len(table) * 1.2), 4))
axis.bar(table["run"], table["validation_accuracy"], color=["#b5651d" if t == "baseline" else "#4c8c4a" for t in table["tag"]])
axis.set(title=f"{args.task.title()}: baseline vs hyperparameter tuning", ylabel="Validation accuracy", ylim=(0, 1))
axis.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(output / "baseline_vs_tuned.png", dpi=160); plt.close(fig)
print(f"Saved {output / 'experiment_comparison.csv'} and baseline_vs_tuned.png")
