"""
Training and evaluation pipeline for SignBridge.
Trains the neural network model, tracks loss/accuracy curves, and
exports confusion matrices and classification reports for coursework submission.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Ensure src modules are imported
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLASSES, NUM_CLASSES, INPUT_FEATURE_DIM, SEQUENCE_LENGTH,
    BATCH_SIZE, HIDDEN_DIM, NUM_LAYERS, DROPOUT, LEARNING_RATE,
    EPOCHS, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
)
from model import SignBridgeBiLSTM, SignBridgeGRU
from dataset import get_dataloaders


def train_model(
    model_type: str = "bilstm",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE
):
    print(f"=== Starting SignBridge Model Training ({model_type.upper()}) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    train_loader, val_loader = get_dataloaders(
        processed_dir=PROCESSED_DATA_DIR,
        classes=CLASSES,
        batch_size=batch_size,
        val_split=0.2
    )

    if train_loader is None or len(train_loader) == 0:
        print("\n[ERROR] No processed keypoint data found!")
        print(f"Please populate '{PROCESSED_DATA_DIR}' with processed .npy gesture files.")
        print("You can extract from INCLUDE videos or record gestures with collect_data.py first.")
        return

    # Instantiate model
    if model_type.lower() == "bilstm":
        model = SignBridgeBiLSTM(
            input_dim=INPUT_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT
        )
    else:
        model = SignBridgeGRU(
            input_dim=INPUT_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT
        )

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_loss = float("inf")
    best_model_path = MODELS_DIR / f"signbridge_{model_type}.pth"

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
        )

        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": CLASSES,
                "val_accuracy": epoch_val_acc
            }, best_model_path)

    print(f"\n[INFO] Best model saved to: {best_model_path}")

    # Generate Learning Curves Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="blue")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="blue")
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="green", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy Curve")
    plt.legend()
    plt.grid(True)

    curves_path = REPORTS_DIR / "learning_curves.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[INFO] Learning curves saved to: {curves_path}")

    # Final Evaluation & Confusion Matrix
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Classification Report
    labels_idx = list(range(len(CLASSES)))
    report = classification_report(
        all_targets,
        all_preds,
        labels=labels_idx,
        target_names=CLASSES,
        zero_division=0
    )
    print("\n=== Validation Classification Report ===")
    print(report)

    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds, labels=labels_idx)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Confusion Matrix - SignBridge ISL Recognition")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    train_model(model_type="bilstm", epochs=30)
