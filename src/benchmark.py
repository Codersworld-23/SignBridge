"""
SignBridge Experimental Benchmark & Ablation Study Suite.
Trains, evaluates, and exports results into 4 dedicated individual model folders:
- results/1_bilstm/
- results/2_cnngru/
- results/3_tcn/
- results/4_transformer/
- results/comparison/

Each model folder contains:
1. checkpoint_best.pth (model weights)
2. learning_curves.png (epoch loss and accuracy)
3. confusion_matrix.png (test set confusion matrix)
4. classification_report.txt (precision, recall, f1 per class)
5. test_metrics.json (all SLR and SLT metrics on held-out test set)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from config import CLASSES, NUM_CLASSES, INPUT_FEATURE_DIM, PROCESSED_DATA_DIR, RESULTS_DIR
from model import get_model_by_name
from dataset import get_all_dataloaders
from metrics import evaluate_slr_slt

# Define the 4 model specifications
MODEL_SPECS = [
    {"name": "bilstm", "display_name": "Bi-LSTM (Baseline)", "folder": RESULTS_DIR / "1_bilstm"},
    {"name": "cnngru", "display_name": "CNN-BiGRU (Local-Global)", "folder": RESULTS_DIR / "2_cnngru"},
    {"name": "tcn", "display_name": "Res-TCN (Dilated Temporal)", "folder": RESULTS_DIR / "3_tcn"},
    {"name": "transformer", "display_name": "Transformer (Self-Attention)", "folder": RESULTS_DIR / "4_transformer"},
]

COMPARISON_DIR = RESULTS_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
for spec in MODEL_SPECS:
    spec["folder"].mkdir(parents=True, exist_ok=True)


def train_single_model(
    model_name: str,
    output_dir: Path,
    train_loader,
    val_loader,
    input_dim: int = INPUT_FEATURE_DIM,
    num_classes: int = NUM_CLASSES,
    epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  Training Architecture: {model_name.upper()}")
    print(f"  Target Results Folder: {output_dir}")
    print(f"=======================================================")

    model = get_model_by_name(model_name, input_dim=input_dim, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_checkpoint = output_dir / "checkpoint_best.pth"

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        t_loss = running_loss / max(total, 1)
        t_acc = correct / max(total, 1)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_name.upper()}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {t_loss:.4f} (Acc: {t_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.1f}%)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint)

    train_duration = time.time() - start_train_time
    print(f"Training completed in {train_duration:.1f}s. Saved -> {best_checkpoint.name}")

    # Plot & Save Individual Model Learning Curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="crimson", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} - Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", color="royalblue")
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Acc", color="green", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name.upper()} - Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curve_path = output_dir / "learning_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()

    return {
        "model_name": model_name,
        "checkpoint_path": str(best_checkpoint),
        "history": history,
        "train_time_sec": round(train_duration, 1)
    }


def evaluate_and_export_model(
    model_name: str,
    output_dir: Path,
    checkpoint_path: str,
    test_loader,
    classes: List[str],
    input_dim: int = INPUT_FEATURE_DIM,
    device: str = "cpu"
) -> Dict[str, Any]:
    print(f"Evaluating {model_name.upper()} on Held-Out Test Set...")
    model = get_model_by_name(model_name, input_dim=input_dim, num_classes=len(classes))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs, all_targets, all_preds = [], [], []
    inference_start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    inference_time = (time.time() - inference_start) / max(len(all_targets), 1) * 1000
    all_probs = np.vstack(all_probs)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # 1. SLR & SLT Metrics
    metrics = evaluate_slr_slt(all_targets, all_probs, classes)
    metrics["Latency_ms"] = round(inference_time, 2)

    # Save test_metrics.json in the model's own folder
    metrics_json_path = output_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Classification Report
    labels_idx = list(range(len(classes)))
    report = classification_report(all_targets, all_preds, labels=labels_idx, target_names=classes, zero_division=0)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 3. Confusion Matrix Plot (top classes for readability)
    top_n = min(30, len(classes))
    top_indices = list(range(top_n))
    mask = np.isin(all_targets, top_indices) & np.isin(all_preds, top_indices)
    if np.any(mask):
        cm = confusion_matrix(all_targets[mask], all_preds[mask], labels=top_indices)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=[classes[i] for i in top_indices], yticklabels=[classes[i] for i in top_indices])
        plt.title(f"{model_name.upper()} - Confusion Matrix (Top {top_n} Classes)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()

    print(f"Results exported to: {output_dir}")
    return metrics


def generate_comparison_artifacts(all_results: List[Dict[str, Any]], all_histories: Dict[str, Dict[str, List[float]]]):
    print(f"\n=== Generating Consolidated Comparison Reports in {COMPARISON_DIR} ===")

    rows = []
    for res in all_results:
        m_name = res["display_name"]
        slr = res["metrics"]["SLR"]
        slt = res["metrics"]["SLT"]
        lat = res["metrics"]["Latency_ms"]

        rows.append({
            "Architecture": m_name,
            "Top-1 Acc (%)": slr["Top1_Accuracy"],
            "Top-5 Acc (%)": slr["Top5_Accuracy"],
            "F1-Score (%)": slr["F1_Macro"],
            "WER (%)": slr["WER"],
            "BLEU-1": slt["BLEU_1"],
            "BLEU-4": slt["BLEU_4"],
            "ROUGE-L": slt["ROUGE_L_F1"],
            "METEOR": slt["METEOR"],
            "PER (%)": slt["PER"],
            "Latency (ms)": lat
        })

    df = pd.DataFrame(rows)

    # 1. Comparison CSV
    df.to_csv(COMPARISON_DIR / "benchmark_comparison.csv", index=False)

    # 2. Comparison Markdown
    with open(COMPARISON_DIR / "benchmark_comparison.md", "w", encoding="utf-8") as f:
        f.write("# SignBridge 4-Model Benchmark Comparison (Held-Out Test Set)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Evaluated strictly on independent 15% held-out test partition.*")

    # 3. Comparison LaTeX Table
    latex_code = df.to_latex(index=False, caption="SignBridge Multi-Model Benchmark on Held-Out Test Set", label="tab:signbridge_benchmark")
    with open(COMPARISON_DIR / "benchmark_comparison.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    # 4. Comparative Learning Curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for m_name, hist in all_histories.items():
        plt.plot(hist["val_loss"], label=f"{m_name.upper()} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Comparative Validation Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for m_name, hist in all_histories.items():
        plt.plot([a * 100 for a in hist["val_acc"]], label=f"{m_name.upper()} Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Comparative Validation Accuracy Curves")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "comparison_learning_curves.png", dpi=300)
    plt.close()

    # 5. Comparative Metric Bar Chart
    plt.figure(figsize=(10, 6))
    bar_df = df.melt(id_vars=["Architecture"], value_vars=["Top-1 Acc (%)", "Top-5 Acc (%)", "ROUGE-L", "BLEU-1"], var_name="Metric", value_name="Score")
    sns.barplot(data=bar_df, x="Metric", y="Score", hue="Architecture", palette="tab10")
    plt.title("Comparative Performance across SLR & SLT Metrics")
    plt.ylabel("Percentage / Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "comparison_metrics_bar.png", dpi=300)
    plt.close()

    print(f"Consolidated comparison tables and plots exported to: {COMPARISON_DIR}")
    print("\n" + df.to_string(index=False))


def run_benchmark_pipeline(epochs: int = 15, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting SignBridge 4-Model Pipeline on {device.upper()} ===")

    manifest_file = RESULTS_DIR / "split_manifest.json"
    train_loader, val_loader, test_loader = get_all_dataloaders(
        processed_dir=PROCESSED_DATA_DIR,
        classes=CLASSES,
        manifest_path=manifest_file,
        batch_size=batch_size,
        seed=42
    )

    if train_loader is None:
        print("[ERROR] No data found in processed directory!")
        return

    all_results = []
    all_histories = {}

    for spec in MODEL_SPECS:
        m_name = spec["name"]
        m_display = spec["display_name"]
        m_folder = spec["folder"]

        # Train & validate
        train_res = train_single_model(
            model_name=m_name,
            output_dir=m_folder,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=INPUT_FEATURE_DIM,
            num_classes=NUM_CLASSES,
            epochs=epochs,
            device=device
        )
        all_histories[m_name] = train_res["history"]

        # Evaluate on Test Set & export directly into model folder
        test_metrics = evaluate_and_export_model(
            model_name=m_name,
            output_dir=m_folder,
            checkpoint_path=train_res["checkpoint_path"],
            test_loader=test_loader,
            classes=CLASSES,
            input_dim=INPUT_FEATURE_DIM,
            device=device
        )

        all_results.append({
            "model_name": m_name,
            "display_name": m_display,
            "folder": str(m_folder),
            "metrics": test_metrics,
            "train_time_sec": train_res["train_time_sec"]
        })

    # Consolidated comparison
    generate_comparison_artifacts(all_results, all_histories)


if __name__ == "__main__":
    run_benchmark_pipeline(epochs=15, batch_size=32)
