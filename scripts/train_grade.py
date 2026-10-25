import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_or_build_manifest
from src.training import run_training

parser = argparse.ArgumentParser(description="Train a tea-grade classifier.")
parser.add_argument("--model", default="efficientnet", choices=["efficientnet", "mobilenet", "resnet", "shufflenet", "squeezenet"])
parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--image-size", type=int, default=224, help="Use 160 for faster exploratory Colab runs.")
parser.add_argument("--workers", type=int, default=None, help="Data-loading workers; defaults to 2 on a GPU.")
parser.add_argument("--patience", type=int, default=5, help="Stop after this many non-improving validation epochs.")
args = parser.parse_args()
run_dir, config = run_training(load_or_build_manifest(), "grade", args.model, args.epochs, args.batch_size, args.lr, args.weight_decay,
                               image_size=args.image_size, num_workers=args.workers, early_stopping_patience=args.patience)
print(f"Finished. Results: {run_dir}\nBest validation accuracy: {config['validation_accuracy']:.3f}")
