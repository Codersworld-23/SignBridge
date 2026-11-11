"""
Converts pre-extracted MediaPipe Holistic Parquet files from the Kaggle INCLUDE archive
into SignBridge's standardized (30, 258) .npy landmark sequences for training.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Path configurations
PROJECT_ROOT = Path(r"C:\Users\kiran\OneDrive\Desktop\sem7\Fuzzy\SignBridge")
ARCHIVE_KEYPOINTS = PROJECT_ROOT / "raw_dataset" / "archive" / "keypoints"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Candidate 20 high-value conversational words available directly in INCLUDE
DATASET_WORD_MAP = {
    # Label Name : (category, subfolder_in_archive)
    "Hello": ("greetings", "hello"),
    "Thank You": ("greetings", "thankyou"),
    "How Are You": ("greetings", "howareyou"),
    "Good Morning": ("greetings", "goodmorning"),
    "Good Night": ("greetings", "goodnight"),
    "Pleased": ("greetings", "pleased"),
    "Alright": ("greetings", "alright"),
    "I": ("pronouns", "i"),
    "You": ("pronouns", "you"),
    "We": ("pronouns", "we"),
    "Good": ("adjectives", "good"),
    "Bad": ("adjectives", "bad"),
    "Happy": ("adjectives", "happy"),
    "Sad": ("adjectives", "sad"),
    "Hospital": ("places", "hospital"),
    "House": ("places", "house"),
    "Doctor": ("jobs", "doctor"),
    "Patient": ("jobs", "patient"),
    "Father": ("people", "father"),
    "Cellphone": ("electronics", "cellphone")
}


def parquet_to_landmarks(parquet_path: Path, target_frames: int = 30) -> np.ndarray:
    """
    Parses a single Parquet file and extracts a (target_frames, 258) array:
    - Pose: 33 * 4 (x, y, z, visibility) = 132
    - Left Hand: 21 * 3 (x, y, z) = 63
    - Right Hand: 21 * 3 (x, y, z) = 63
    Total = 258 features per frame.
    """
    df = pd.read_parquet(parquet_path)
    frames = sorted(df["frame"].unique())
    if len(frames) == 0:
        return None

    raw_sequence = []
    for f in frames:
        f_df = df[df["frame"] == f]

        # Pose (33 landmarks)
        pose_df = f_df[f_df["type"] == "pose"].sort_values("landmark_index")
        if len(pose_df) == 33:
            xyz = pose_df[["x", "y", "z"]].fillna(0.0).values
            vis = (~pose_df["x"].isna()).astype(float).values[:, None]
            pose_feat = np.hstack([xyz, vis]).flatten()
        else:
            pose_feat = np.zeros(33 * 4, dtype=np.float32)

        # Left Hand (21 landmarks)
        lh_df = f_df[f_df["type"] == "left_hand"].sort_values("landmark_index")
        if len(lh_df) == 21:
            lh_feat = lh_df[["x", "y", "z"]].fillna(0.0).values.flatten()
        else:
            lh_feat = np.zeros(21 * 3, dtype=np.float32)

        # Right Hand (21 landmarks)
        rh_df = f_df[f_df["type"] == "right_hand"].sort_values("landmark_index")
        if len(rh_df) == 21:
            rh_feat = rh_df[["x", "y", "z"]].fillna(0.0).values.flatten()
        else:
            rh_feat = np.zeros(21 * 3, dtype=np.float32)

        frame_vec = np.concatenate([pose_feat, lh_feat, rh_feat]).astype(np.float32)
        raw_sequence.append(frame_vec)

    raw_sequence = np.array(raw_sequence, dtype=np.float32)
    T_orig = len(raw_sequence)

    # Uniform temporal resampling to target_frames
    if T_orig == target_frames:
        return raw_sequence

    indices = np.linspace(0, T_orig - 1, target_frames)
    resampled = np.zeros((target_frames, 258), dtype=np.float32)
    for dim in range(258):
        resampled[:, dim] = np.interp(indices, np.arange(T_orig), raw_sequence[:, dim])

    return resampled


def convert_all_classes():
    print("=== Converting Kaggle Parquet Files to SignBridge Training Data ===")
    total_converted = 0

    for label_name, (cat, folder_name) in DATASET_WORD_MAP.items():
        src_dir = ARCHIVE_KEYPOINTS / cat / folder_name
        dest_dir = PROCESSED_DATA_DIR / label_name

        # Ensure destination directory exists and is clean
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Clear synthetic dummy samples if present
        for old_f in dest_dir.glob("sample_*.npy"):
            old_f.unlink()

        if not src_dir.exists():
            print(f"[WARNING] Source folder not found: {src_dir}")
            continue

        parquet_files = list(src_dir.glob("*.parquet"))
        print(f"Converting '{label_name}': {len(parquet_files)} files...")

        count = 0
        for idx, p_file in enumerate(parquet_files):
            try:
                arr = parquet_to_landmarks(p_file, target_frames=30)
                if arr is not None:
                    out_path = dest_dir / f"sample_{idx:04d}.npy"
                    np.save(out_path, arr)
                    count += 1
                    total_converted += 1
            except Exception as e:
                print(f"  Error on {p_file.name}: {e}")

        print(f"  -> Converted {count} samples for '{label_name}'")

    print(f"\n[DONE] Successfully converted {total_converted} real samples into {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    convert_all_classes()
