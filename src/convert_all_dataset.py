"""
High-performance batch converter for the full INCLUDE dataset.
Converts all 4,284 pre-extracted MediaPipe Holistic Parquet files across 262 classes
into normalized (30, 258) .npy landmark sequences using multi-threading.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\kiran\OneDrive\Desktop\sem7\Fuzzy\SignBridge")
ARCHIVE_KEYPOINTS = PROJECT_ROOT / "raw_dataset" / "archive" / "keypoints"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def process_single_parquet(parquet_path: Path, dest_npy_path: Path, target_frames: int = 30) -> bool:
    """
    Parses a single Parquet file and saves a (target_frames, 258) array:
    - Pose: 33 * 4 (x, y, z, visibility) = 132
    - Left Hand: 21 * 3 (x, y, z) = 63
    - Right Hand: 21 * 3 (x, y, z) = 63
    Total = 258 features per frame.
    """
    try:
        df = pd.read_parquet(parquet_path)
        frames = sorted(df["frame"].unique())
        if len(frames) == 0:
            return False

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

        # Uniform temporal resampling to target_frames (30)
        if T_orig == target_frames:
            resampled = raw_sequence
        else:
            indices = np.linspace(0, T_orig - 1, target_frames)
            resampled = np.zeros((target_frames, 258), dtype=np.float32)
            for dim in range(258):
                resampled[:, dim] = np.interp(indices, np.arange(T_orig), raw_sequence[:, dim])

        np.save(dest_npy_path, resampled)
        return True
    except Exception as e:
        return False


def run_full_conversion(max_workers: int = 8):
    start_time = time.time()
    print("=== Scanning Full INCLUDE Dataset for Conversion ===")

    # Discover all parquet files across all categories and classes
    conversion_tasks = []
    classes_discovered = set()

    for cat_dir in ARCHIVE_KEYPOINTS.iterdir():
        if not cat_dir.is_dir():
            continue
        for class_dir in cat_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            classes_discovered.add(class_name)

            dest_class_dir = PROCESSED_DATA_DIR / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            parquet_files = sorted(class_dir.glob("*.parquet"))
            for idx, p_path in enumerate(parquet_files):
                dest_npy = dest_class_dir / f"sample_{idx:04d}.npy"
                # If already converted and valid, skip
                if not dest_npy.exists():
                    conversion_tasks.append((p_path, dest_npy))

    total_classes = len(classes_discovered)
    print(f"Discovered {total_classes} unique sign classes.")
    print(f"Total files to process: {len(conversion_tasks)}")

    if len(conversion_tasks) == 0:
        print("[INFO] All files are already converted!")
        return

    print(f"Starting multi-threaded conversion using {max_workers} worker threads...")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_parquet, p_path, dest_npy): (p_path, dest_npy)
            for p_path, dest_npy in conversion_tasks
        }

        for future in as_completed(futures):
            if future.result():
                completed += 1
            if completed % 500 == 0 or completed == len(conversion_tasks):
                elapsed = time.time() - start_time
                print(f"Progress: {completed}/{len(conversion_tasks)} files converted ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n[DONE] Full conversion completed in {total_time:.2f} seconds.")
    print(f"Dataset ready at: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    run_full_conversion(max_workers=8)
