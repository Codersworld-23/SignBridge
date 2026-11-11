"""
PyTorch Dataset and DataLoaders for SignBridge keypoint sequences.
Handles reading processed .npy files, data augmentation, and
enforces a strict Train (70%) / Val (15%) / Test (15%) experimental split.
"""

import os
import json
import glob
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SignLanguageDataset(Dataset):
    """
    Dataset for temporal sign language keypoint sequences.
    Each sample is a NumPy array of shape (30, 258), label is an int.
    """

    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        augment: bool = False
    ):
        self.samples = samples
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, sequence: np.ndarray) -> np.ndarray:
        """Lightweight spatial Gaussian jitter + temporal scaling augmentation."""
        augmented = sequence.copy()
        noise = np.random.normal(0, 0.006, augmented.shape).astype(np.float32)
        augmented += noise

        # Random temporal speed variation
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.92, 1.08)
            orig_len = len(augmented)
            resampled = np.zeros_like(augmented)
            indices = np.linspace(0, orig_len - 1, orig_len)
            for d in range(augmented.shape[1]):
                resampled[:, d] = np.interp(indices, np.linspace(0, orig_len - 1, orig_len), augmented[:, d])
            augmented = resampled

        return augmented.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        if self.augment:
            sample = self._apply_augmentation(sample)

        x_tensor = torch.tensor(sample, dtype=torch.float32)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_tensor, y_tensor


def load_dataset_with_manifest(
    processed_dir: Path,
    classes: List[str],
    manifest_path: Optional[Path] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Tuple[List[np.ndarray], List[int]]]:
    """
    Loads all samples and partitions them into Train, Validation, and Test splits.
    Saves/loads partition manifest to guarantee a strict, reproducible held-out test set.
    """
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

    all_file_records = []
    for cls_name, cls_id in class_to_id.items():
        cls_folder = processed_dir / cls_name
        if not cls_folder.exists():
            continue

        for f_path in cls_folder.glob("*.npy"):
            all_file_records.append({
                "path": str(f_path),
                "class_name": cls_name,
                "label": cls_id
            })

    if not all_file_records:
        return {"train": ([], []), "val": ([], []), "test": ([], [])}

    # Group by class to ensure stratified representation
    np.random.seed(seed)
    class_groups: Dict[int, List[Dict[str, Any]]] = {}
    for rec in all_file_records:
        class_groups.setdefault(rec["label"], []).append(rec)

    train_recs, val_recs, test_recs = [], [], []

    for label, items in class_groups.items():
        np.random.shuffle(items)
        n = len(items)
        if n == 1:
            train_recs.append(items[0])
        elif n == 2:
            train_recs.append(items[0])
            val_recs.append(items[1])
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            train_recs.extend(items[:n_train])
            val_recs.extend(items[n_train:n_train + n_val])
            test_recs.extend(items[n_train + n_val:])

    # Save manifest for coursework documentation
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "total_samples": len(all_file_records),
            "train_count": len(train_recs),
            "val_count": len(val_recs),
            "test_count": len(test_recs),
            "test_files": [r["path"] for r in test_recs]
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def load_arrays(recs):
        samples, labels = [], []
        for r in recs:
            try:
                arr = np.load(r["path"])
                if arr.ndim == 2 and arr.shape[1] == 258:
                    samples.append(arr.astype(np.float32))
                    labels.append(r["label"])
            except Exception:
                pass
        return samples, labels

    return {
        "train": load_arrays(train_recs),
        "val": load_arrays(val_recs),
        "test": load_arrays(test_recs)
    }


def get_all_dataloaders(
    processed_dir: Path,
    classes: List[str],
    manifest_path: Optional[Path] = None,
    batch_size: int = 16,
    seed: int = 42
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Returns (train_loader, val_loader, test_loader) with strict train/val/test segregation.
    """
    splits = load_dataset_with_manifest(
        processed_dir=processed_dir,
        classes=classes,
        manifest_path=manifest_path,
        seed=seed
    )

    train_samples, train_labels = splits["train"]
    val_samples, val_labels = splits["val"]
    test_samples, test_labels = splits["test"]

    if len(train_samples) == 0:
        return None, None, None

    train_ds = SignLanguageDataset(train_samples, train_labels, augment=True)
    val_ds = SignLanguageDataset(val_samples, val_labels, augment=False)
    test_ds = SignLanguageDataset(test_samples, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
