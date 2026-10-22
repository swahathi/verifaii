"""
Dataset Loader & Preprocessing
Inspects dataset structure dynamically without assuming rigid folder names.
Splits into Train, Validation, and Test sets BEFORE fitting transformers (preventing data leakage).
"""

import os
import glob
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from metadata_extractor import extract_metadata
from visual_feature_extractor import extract_visual_features


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def scan_dataset_directory(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scans data directory dynamically.
    Detects classes from folder names (e.g. real/ai, authentic/generated, 0/1).
    """
    records = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if len(subdirs) >= 2:
        # Subfolder structure
        for subdir in subdirs:
            folder_path = os.path.join(data_dir, subdir)
            lower_name = subdir.lower()
            # Determine label
            if any(k in lower_name for k in ["real", "authentic", "photo", "original", "0"]):
                label = 0  # Real
                class_name = "Real"
            elif any(k in lower_name for k in ["ai", "fake", "generated", "synthetic", "1"]):
                label = 1  # AI-Generated
                class_name = "AI-Generated"
            else:
                label = 0 if subdir == subdirs[0] else 1
                class_name = subdir

            files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(SUPPORTED_EXTENSIONS)
            ]
            for f in files:
                records.append({"file_path": f, "label": label, "class_name": class_name})
    else:
        # Flat folder with CSV or search
        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        for f in files:
            name = os.path.basename(f).lower()
            label = 1 if "ai" in name or "fake" in name else 0
            records.append({"file_path": f, "label": label, "class_name": "AI" if label == 1 else "Real"})

    print(f"[Dataset Loader] Found {len(records)} images in '{data_dir}'.")
    return records


def extract_features_from_dataset(records: List[Dict[str, Any]], max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Extracts metadata and visual features for all records.
    Returns structured Pandas DataFrame.
    """
    if max_samples and len(records) > max_samples:
        import random
        random.seed(42)
        records = random.sample(records, max_samples)

    rows = []
    print(f"[Dataset Loader] Extracting features from {len(records)} images...")

    for idx, rec in enumerate(records):
        path = rec["file_path"]
        label = rec["label"]

        # Extract features
        meta = extract_metadata(path)
        visual = extract_visual_features(path)

        combined = {
            "file_path": path,
            "label": label,
            # Metadata features
            "has_exif": meta["has_exif"],
            "has_camera_make": meta["has_camera_make"],
            "has_camera_model": meta["has_camera_model"],
            "has_datetime_original": meta["has_datetime_original"],
            "has_software": meta["has_software"],
            "software_is_ai": meta["software_is_ai"],
            # Visual features
            "brightness": visual["brightness"],
            "contrast": visual["contrast"],
            "sharpness": visual["sharpness"],
            "edge_density": visual["edge_density"],
            "color_saturation": visual["color_saturation"],
            "color_dispersion": visual["color_dispersion"],
            "texture_noise_residual": visual["texture_noise_residual"],
            "image_entropy": visual["image_entropy"],
        }
        rows.append(combined)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(records):
            print(f"  Processed {idx + 1}/{len(records)} images...")

    df = pd.DataFrame(rows)
    return df


def prepare_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leak-free stratified 70/15/15 train/val/test split.
    """
    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    # Then split train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["label"],
        random_state=random_state
    )

    print(f"[Dataset Split] Stratified Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df
