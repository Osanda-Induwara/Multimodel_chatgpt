"""Shared training/evaluation helpers used by the small task scripts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import RESULTS_ROOT, SEED
from .data import TeaDataset, grouped_split
from .models import create_model

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def transforms_for_training(image_size: int):
    train = transforms.Compose([
        transforms.RandomRotation(15), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.08),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=.20, scale=(.02, .10)),
    ])
    valid = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, valid


def make_run_directory(task: str, model: str, tag: str):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_ROOT / task / f"{stamp}_{model}_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_history(history, path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["valid_loss"], label="validation")
    axes[0].set(title="Learning curve", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["valid_accuracy"], label="validation")
    axes[1].set(title="Epoch vs accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_confusion(true, predicted, labels, path: Path):
    matrix = confusion_matrix(true, predicted, labels=range(len(labels)))
    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrBr", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set(xlabel="Predicted", ylabel="Actual", title="Validation confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def run_training(frame: pd.DataFrame, task: str, model_name: str, epochs=20, batch_size=16,
                 learning_rate=3e-4, weight_decay=1e-4, image_size=224, tag="baseline", seed=SEED,
                 num_workers: int | None = None, early_stopping_patience: int = 5):
    torch.manual_seed(seed); np.random.seed(seed)
    run_dir = make_run_directory(task, model_name, tag)
    train_frame, valid_frame = grouped_split(frame, seed=seed)
    train_transform, valid_transform = transforms_for_training(image_size)
    train_data = TeaDataset(train_frame, task, train_transform, image_size)
    valid_data = TeaDataset(valid_frame, task, valid_transform, image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, len(train_data.labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    if num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0
    loader_options = {"num_workers": num_workers, "pin_memory": device.type == "cuda",
                      "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, **loader_options)
    history = {key: [] for key in ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]}
    best_accuracy, best_state = -1.0, None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train(); running_loss = correct = count = 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(); output = model(images); loss = criterion(output, targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * len(targets); correct += (output.argmax(1) == targets).sum().item(); count += len(targets)
        history["train_loss"].append(running_loss / count); history["train_accuracy"].append(correct / count)
        valid_loss, valid_accuracy, _, _ = evaluate(model, valid_loader, criterion, device)
        history["valid_loss"].append(valid_loss); history["valid_accuracy"].append(valid_accuracy)
        print(f"epoch {epoch + 1:02d}/{epochs}: train_acc={history['train_accuracy'][-1]:.3f}, valid_acc={valid_accuracy:.3f}")
        if valid_accuracy > best_accuracy:
            best_accuracy, best_state = valid_accuracy, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            # Save immediately: a Colab interruption still preserves the best model.
            torch.save({"state_dict": best_state, "labels": train_data.labels, "task": task,
                        "model": model_name, "epoch": epoch + 1, "validation_accuracy": best_accuracy},
                       run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs (no validation improvement for {early_stopping_patience} epochs).")
            break

    model.load_state_dict(best_state)
    _, best_accuracy, truth, predictions = evaluate(model, valid_loader, criterion, device)
    torch.save({"state_dict": model.state_dict(), "labels": train_data.labels, "task": task, "model": model_name}, run_dir / "best_model.pt")
    plot_history(history, run_dir / "learning_curves.png")
    plot_confusion(truth, predictions, train_data.labels, run_dir / "confusion_matrix.png")
    report = classification_report(truth, predictions, target_names=train_data.labels, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(run_dir / "classification_report.csv")
    pd.DataFrame({"path": valid_frame.path, "actual": [train_data.labels[x] for x in truth],
                  "predicted": [train_data.labels[x] for x in predictions]}).to_csv(run_dir / "validation_predictions.csv", index=False)
    config = {"task": task, "model": model_name, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "weight_decay": weight_decay, "image_size": image_size,
              "validation_accuracy": best_accuracy, "train_images": len(train_frame), "validation_images": len(valid_frame),
              "train_samples": train_frame.sample_id.nunique(), "validation_samples": valid_frame.sample_id.nunique(), "tag": tag,
              "early_stopping_patience": early_stopping_patience, "epochs_completed": len(history["train_loss"])}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir, config


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_total = correct = count = 0; truth, predictions = [], []
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images); loss_total += criterion(output, targets).item() * len(targets)
        predicted = output.argmax(1); correct += (predicted == targets).sum().item(); count += len(targets)
        truth.extend(targets.cpu().tolist()); predictions.extend(predicted.cpu().tolist())
    return loss_total / count, correct / count, truth, predictions
