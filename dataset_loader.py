"""
Dataset Loader & Preprocessing Module
Inspects dataset structure dynamically without assuming rigid folder names.
Supports nested subdirectories and common image formats.
Splits into Train, Validation, and Test sets BEFORE fitting transformers (preventing data leakage).
"""

import os
import re
import random
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from metadata_extractor import extract_metadata
from visual_feature_extractor import extract_visual_features


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")

# Patterns to identify AI-generated image directories (label = 1)
AI_PATTERNS = [
    r"\bai\b",
    r"fake",
    r"synthetic",
    r"generated",
    r"deepfake",
    r"synth",
    r"midjourney",
    r"dall-?e",
    r"stable[-_]?diffusion",
    r"genai",
]

# Patterns to identify authentic/real image directories (label = 0)
REAL_PATTERNS = [
    r"\breal\b",
    r"authentic",
    r"original",
    r"photo",
    r"natural",
    r"camera",
]


def _match_class_label(folder_name: str) -> Optional[int]:
    """
    Determines whether a folder name corresponds to Real (0) or AI-Generated (1).
    Uses regex boundary matching to avoid accidental false matches (e.g. 'train' or 'portrait').
    """
    lower = folder_name.lower().strip()

    # Check AI patterns first
    for pat in AI_PATTERNS:
        if re.search(pat, lower):
            return 1

    # Check Real patterns
    for pat in REAL_PATTERNS:
        if re.search(pat, lower):
            return 0

    return None


def scan_dataset_directory(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scans data directory dynamically, supporting nested subfolders and common image formats.
    Correctly identifies classes: Real = 0, AI-Generated = 1.
    Prevents crashes if classes or images are missing and reports class counts.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    records: List[Dict[str, Any]] = []

    # Check immediate subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if subdirs:
        # Traverse subdirectories
        for subdir in subdirs:
            folder_path = os.path.join(data_dir, subdir)
            label = _match_class_label(subdir)

            # If top-level subdir does not match directly, check if it's a split folder (train/test/val)
            # or traverse nested subfolders
            if label is None:
                # Walk down to find nested class folders
                for root, dirs, files in os.walk(folder_path):
                    root_name = os.path.basename(root)
                    nested_label = _match_class_label(root_name)
                    if nested_label is not None:
                        for f in files:
                            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                                records.append({
                                    "file_path": os.path.join(root, f),
                                    "label": nested_label,
                                    "class_name": "AI-Generated" if nested_label == 1 else "Real",
                                })
            else:
                # Walk all files in this class folder, including any nested subdirectories
                for root, _, files in os.walk(folder_path):
                    for f in files:
                        if f.lower().endswith(SUPPORTED_EXTENSIONS):
                            records.append({
                                "file_path": os.path.join(root, f),
                                "label": label,
                                "class_name": "AI-Generated" if label == 1 else "Real",
                            })

    # Fallback: if no subdirectories had images, check flat folder
    if not records:
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    fname = f.lower()
                    label = 1 if any(re.search(pat, fname) for pat in AI_PATTERNS) else 0
                    records.append({
                        "file_path": os.path.join(root, f),
                        "label": label,
                        "class_name": "AI-Generated" if label == 1 else "Real",
                    })

    if not records:
        raise FileNotFoundError(f"No valid image files found in '{data_dir}' with extensions {SUPPORTED_EXTENSIONS}.")

    # Count samples per class
    real_count = sum(1 for r in records if r["label"] == 0)
    ai_count = sum(1 for r in records if r["label"] == 1)

    print(f"[Dataset Loader] Found {len(records)} images in '{data_dir}':")
    print(f"  - Real Photographs (label 0): {real_count}")
    print(f"  - AI-Generated Images (label 1): {ai_count}")

    # Validate that both classes are present
    if real_count == 0 or ai_count == 0:
        raise ValueError(
            f"Class missing in dataset! Detected: Real={real_count}, AI-Generated={ai_count}. "
            f"Both classes (Real=0 and AI-Generated=1) are required for training."
        )

    return records


def extract_features_from_dataset(
    records: List[Dict[str, Any]],
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Extracts metadata and visual features for dataset records.
    Applies balanced sampling across classes when max_samples is provided to prevent class imbalance.
    """
    # Balanced sampling across classes if max_samples is specified
    if max_samples and len(records) > max_samples:
        real_records = [r for r in records if r["label"] == 0]
        ai_records = [r for r in records if r["label"] == 1]

        target_per_class = max_samples // 2
        rng = random.Random(42)

        sampled_real = rng.sample(real_records, min(len(real_records), target_per_class))
        sampled_ai = rng.sample(ai_records, min(len(ai_records), target_per_class))

        records = sampled_real + sampled_ai
        rng.shuffle(records)

        print(
            f"[Dataset Loader] Applied class-balanced sampling (max_samples={max_samples}): "
            f"{len(sampled_real)} Real + {len(sampled_ai)} AI = {len(records)} total."
        )

    rows = []
    total = len(records)
    print(f"[Dataset Loader] Extracting features from {total} images...")

    for idx, rec in enumerate(records):
        path = rec["file_path"]
        label = rec["label"]

        try:
            meta = extract_metadata(path)
            visual = extract_visual_features(path)

            combined = {
                "file_path": path,
                "label": int(label),
                # Metadata features
                "has_exif": int(meta.get("has_exif", 0)),
                "has_camera_make": int(meta.get("has_camera_make", 0)),
                "has_camera_model": int(meta.get("has_camera_model", 0)),
                "has_datetime_original": int(meta.get("has_datetime_original", 0)),
                "has_software": int(meta.get("has_software", 0)),
                "software_is_ai": int(meta.get("software_is_ai", 0)),
                # Visual features
                "brightness": float(visual.get("brightness", 0.5)),
                "contrast": float(visual.get("contrast", 0.5)),
                "sharpness": float(visual.get("sharpness", 0.5)),
                "edge_density": float(visual.get("edge_density", 0.2)),
                "color_saturation": float(visual.get("color_saturation", 0.4)),
                "color_dispersion": float(visual.get("color_dispersion", 0.2)),
                "texture_noise_residual": float(visual.get("texture_noise_residual", 0.25)),
                "image_entropy": float(visual.get("image_entropy", 7.0)),
            }
            rows.append(combined)
        except Exception as e:
            # Skip corrupted images gracefully
            continue

        if (idx + 1) % 250 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} images...")

    df = pd.DataFrame(rows)
    print(f"[Dataset Loader] Successfully extracted features for {len(df)} images.")
    return df


def prepare_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leak-free stratified 70/15/15 train/val/test split.
    Guarantees consistent class proportions across all subsets.
    """
    if len(df["label"].unique()) < 2:
        raise ValueError("Cannot perform stratified split: DataFrame must contain both classes (0 and 1).")

    # First split off held-out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # Then split train and validation sets
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    print(
        f"[Dataset Split] Stratified Splits: "
        f"Train={len(train_df)} (Real: {sum(train_df['label']==0)}, AI: {sum(train_df['label']==1)}), "
        f"Val={len(val_df)} (Real: {sum(val_df['label']==0)}, AI: {sum(val_df['label']==1)}), "
        f"Test={len(test_df)} (Real: {sum(test_df['label']==0)}, AI: {sum(test_df['label']==1)})"
    )
    return train_df, val_df, test_df
