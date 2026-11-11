"""
Pipeline Verification and Smoke Test for SignBridge.
Tests:
1. Fuzzy Inference System (Mamdani evaluation & defuzzification)
2. PyTorch Bi-LSTM neural model forward pass
3. Generates small synthetic sample data for pipeline dry-runs
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from config import (
    CLASSES, INPUT_FEATURE_DIM, SEQUENCE_LENGTH,
    PROCESSED_DATA_DIR, MODELS_DIR
)
from fuzzy_engine import FuzzySignStabilizer
from model import SignBridgeBiLSTM, SignBridgeGRU


def test_fuzzy_system():
    print("--- [1/3] Testing Fuzzy Inference System (FIS) ---")
    fis = FuzzySignStabilizer()

    test_cases = [
        {"name": "High Conf, Stable, Still", "conf": 0.95, "stab": 0.90, "motion": 0.05, "expected": "COMMIT"},
        {"name": "Medium Conf, Transitioning", "conf": 0.60, "stab": 0.50, "motion": 0.45, "expected": "HOLD"},
        {"name": "Low Conf / Noise", "conf": 0.20, "stab": 0.10, "motion": 0.80, "expected": "REJECT"},
    ]

    for tc in test_cases:
        res = fis.evaluate(tc["conf"], tc["stab"], tc["motion"])
        print(f"Test '{tc['name']}':")
        print(f"  Inputs: Conf={tc['conf']}, Stab={tc['stab']}, Motion={tc['motion']}")
        print(f"  Result: Score={res['decision_score']} -> State=[{res['state']}]")
        assert res["state"] == tc["expected"], f"Expected {tc['expected']}, got {res['state']}"
    print("[PASS] Fuzzy System test passed successfully!\n")


def test_neural_models():
    print("--- [2/3] Testing PyTorch Neural Models ---")
    batch_size = 4
    x = torch.randn(batch_size, SEQUENCE_LENGTH, INPUT_FEATURE_DIM)

    # Bi-LSTM
    bilstm = SignBridgeBiLSTM(
        input_dim=INPUT_FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(CLASSES)
    )
    bilstm.eval()
    out_bilstm = bilstm(x)
    print(f"Bi-LSTM output shape: {out_bilstm.shape} (Expected: [{batch_size}, {len(CLASSES)}])")
    assert out_bilstm.shape == (batch_size, len(CLASSES))

    # GRU
    gru = SignBridgeGRU(
        input_dim=INPUT_FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(CLASSES)
    )
    gru.eval()
    out_gru = gru(x)
    print(f"GRU output shape: {out_gru.shape} (Expected: [{batch_size}, {len(CLASSES)}])")
    assert out_gru.shape == (batch_size, len(CLASSES))
    print("[PASS] Neural models test passed successfully!\n")


def generate_synthetic_samples(samples_per_class: int = 5):
    print("--- [3/3] Generating Synthetic Landmark Data for Smoke Testing ---")
    for cls_name in CLASSES:  # Generate for all 20 classes
        cls_dir = PROCESSED_DATA_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            # Create synthetic sequence with class-specific bias
            seq = np.random.normal(0.5, 0.1, (SEQUENCE_LENGTH, INPUT_FEATURE_DIM)).astype(np.float32)
            np.save(cls_dir / f"sample_{i:04d}.npy", seq)

    print(f"[PASS] Synthetic data generated in {PROCESSED_DATA_DIR}\n")


if __name__ == "__main__":
    test_fuzzy_system()
    test_neural_models()
    generate_synthetic_samples(samples_per_class=5)
    print("=== All Pipeline Verification Tests Passed! ===")
