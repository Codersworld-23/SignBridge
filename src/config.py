"""
Configuration settings for SignBridge.
Defines dataset classes, paths, landmark dimensions, and model hyperparameters.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"

for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    RESULTS_DIR, RESULTS_TABLES_DIR, RESULTS_PLOTS_DIR, RESULTS_CHECKPOINTS_DIR, RESULTS_METRICS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Discover all classes from the dataset (sorted alphabetically)
LABEL_MAP_FILE = PROJECT_ROOT / "raw_dataset" / "archive" / "label_map.json"
if LABEL_MAP_FILE.exists():
    import json
    with open(LABEL_MAP_FILE, "r") as f:
        _lmap = json.load(f)
    CLASSES = sorted(list(_lmap.keys()))
else:
    # Fallback to directories present in processed
    _subdirs = [d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and any(d.glob("*.npy"))]
    CLASSES = sorted(_subdirs) if _subdirs else ["Hello", "Thank You", "Good", "Bad", "I"]

NUM_CLASSES = len(CLASSES)
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_LABEL = {idx: name for idx, name in enumerate(CLASSES)}

# Feature Extraction Parameters
# MediaPipe Holistic landmarks per frame:
# - Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
# - Left Hand: 21 landmarks * 3 (x, y, z) = 63
# - Right Hand: 21 landmarks * 3 (x, y, z) = 63
# Total features per frame = 258
POSE_LANDMARKS = 33 * 4
HAND_LANDMARKS = 21 * 3
INPUT_FEATURE_DIM = POSE_LANDMARKS + (HAND_LANDMARKS * 2)  # 258

# Temporal Sequence Settings
SEQUENCE_LENGTH = 30  # Number of consecutive frames per sample (~1 sec at 30 fps)
FRAME_RATE = 30

# Neural Network Hyperparameters
BATCH_SIZE = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 40

# Fuzzy Inference System Parameters
# Cooldown between consecutive word commitments (in seconds or frames)
COMMIT_COOLDOWN_FRAMES = 25
FUZZY_ACCEPT_THRESHOLD = 0.65
FUZZY_HOLD_THRESHOLD = 0.35
