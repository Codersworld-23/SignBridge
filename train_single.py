"""
Interactive Training & Evaluation CLI for SignBridge.
Allows training individual models manually and exporting their results into their dedicated folders:
- results/1_bilstm/
- results/2_cnngru/
- results/3_tcn/
- results/4_transformer/
- results/comparison/
"""

import sys
import argparse
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from config import CLASSES, NUM_CLASSES, INPUT_FEATURE_DIM, PROCESSED_DATA_DIR, RESULTS_DIR
from dataset import get_all_dataloaders
from benchmark import (
    MODEL_SPECS, COMPARISON_DIR, train_single_model,
    evaluate_and_export_model, generate_comparison_artifacts
)


def get_spec(model_name: str):
    name_clean = model_name.lower().replace("-", "").replace("_", "")
    for spec in MODEL_SPECS:
        if spec["name"] in name_clean or name_clean in spec["name"]:
            return spec
    raise ValueError(f"Unknown model '{model_name}'. Choose from: bilstm, cnngru, tcn, transformer")


def main():
    parser = argparse.ArgumentParser(description="SignBridge Individual Model Trainer & Benchmarker")
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        choices=["bilstm", "cnngru", "tcn", "transformer"],
        help="Model architecture to train (bilstm, cnngru, tcn, transformer)"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Training device ('cuda' or 'cpu')")
    parser.add_argument("--compare", action="store_true", help="Generate comparison tables and charts from trained models")
    args = parser.parse_args()

    # Automatically select cuda if available
    import torch
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    if args.compare:
        print("=== Generating Master Comparison from existing model folders ===")
        all_results = []
        all_histories = {}
        for spec in MODEL_SPECS:
            m_folder = spec["folder"]
            metrics_file = m_folder / "test_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                all_results.append({
                    "model_name": spec["name"],
                    "display_name": spec["display_name"],
                    "folder": str(m_folder),
                    "metrics": metrics,
                    "train_time_sec": 0.0
                })
                # Check history
                all_histories[spec["name"]] = {"val_loss": [0], "val_acc": [metrics["SLR"]["Top1_Accuracy"] / 100]}
            else:
                print(f"[NOTE] Model '{spec['name']}' not yet evaluated (missing {metrics_file.name})")

        if all_results:
            generate_comparison_artifacts(all_results, all_histories)
        else:
            print("[ERROR] No trained model results found to compare!")
        return

    spec = get_spec(args.model)
    m_name = spec["name"]
    m_display = spec["display_name"]
    m_folder = spec["folder"]

    print(f"\n=======================================================")
    print(f"  SignBridge Manual Trainer: {m_display}")
    print(f"  Target Folder: {m_folder}")
    print(f"  Total Classes: {NUM_CLASSES} | Device: {device.upper()}")
    print(f"=======================================================")

    manifest_file = RESULTS_DIR / "split_manifest.json"
    train_loader, val_loader, test_loader = get_all_dataloaders(
        processed_dir=PROCESSED_DATA_DIR,
        classes=CLASSES,
        manifest_path=manifest_file,
        batch_size=args.batch_size,
        seed=42
    )

    if train_loader is None:
        print("[ERROR] No data found in data/processed! Please ensure conversion is complete.")
        return

    # 1. Train and save best weights
    train_res = train_single_model(
        model_name=m_name,
        output_dir=m_folder,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=INPUT_FEATURE_DIM,
        num_classes=NUM_CLASSES,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    # 2. Evaluate on Held-out Test Set & export all metrics
    test_metrics = evaluate_and_export_model(
        model_name=m_name,
        output_dir=m_folder,
        checkpoint_path=train_res["checkpoint_path"],
        test_loader=test_loader,
        classes=CLASSES,
        input_dim=INPUT_FEATURE_DIM,
        device=device
    )

    print(f"\n[SUCCESS] Model '{m_name}' training and testing complete!")
    print(f"All artifacts saved inside: {m_folder}")


if __name__ == "__main__":
    main()
