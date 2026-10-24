"""
Model Training & Benchmarking Pipeline
Trains and compares:
1. Random Forest Classifier
2. XGBoost Classifier
3. Support Vector Machine (SVM)
4. Decision Tree Classifier
5. Logistic Regression

Workflow:
1. Leak-free stratified Train (70%) / Validation (15%) / Test (15%) splits.
2. Fits preprocessor strictly on the training set.
3. Tunes / benchmarks classifiers and selects best model using VALIDATION F1-score.
4. Evaluates final selected model on held-out unseen test split.
5. Saves model artifacts, preprocessor, SHAP explainer, and evaluation metrics.
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import shap

from dataset_loader import scan_dataset_directory, extract_features_from_dataset, prepare_stratified_splits

FEATURE_COLS = [
    "has_exif",
    "has_camera_make",
    "has_camera_model",
    "has_datetime_original",
    "has_software",
    "software_is_ai",
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "color_saturation",
    "color_dispersion",
    "texture_noise_residual",
    "image_entropy",
]


def train_and_evaluate(
    data_dir: str = "data",
    output_dir: str = "models",
    max_samples: int = 2000,
) -> Dict[str, Any]:
    """
    Executes the complete training, benchmarking, validation selection,
    and held-out test evaluation pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset & extract features
    print(f"\n--- STEP 1: Scanning Dataset from '{data_dir}' ---")
    records = scan_dataset_directory(data_dir)

    print(f"\n--- STEP 2: Extracting Features (max_samples={max_samples}) ---")
    df = extract_features_from_dataset(records, max_samples=max_samples)

    if len(df) == 0:
        raise RuntimeError("No features could be extracted from dataset. Aborting training.")

    # 2. Stratified train/val/test split to prevent data leakage
    print("\n--- STEP 3: Preparing Stratified Leak-Free Splits ---")
    train_df, val_df, test_df = prepare_stratified_splits(df, test_size=0.15, val_size=0.15, random_state=42)

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    # 3. Fit preprocessor ONLY on training set (no leakage into val/test)
    print("\n--- STEP 4: Fitting Preprocessor on Training Set ---")
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    # Save preprocessor artifact
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[Artifact] Saved preprocessor to '{preprocessor_path}'.")

    # 4. Define candidate classifiers
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42, n_jobs=-1
        ),
        "SVM (RBF)": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, cache_size=500, random_state=42),
            ensemble=False,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=20, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
    }

    trained_models: Dict[str, Any] = {}
    val_metrics: Dict[str, Dict[str, Any]] = {}
    test_metrics: Dict[str, Dict[str, Any]] = {}
    results_records = []

    print("\n" + "=" * 76)
    print("CLASSIFIER BENCHMARKING (TRAIN -> VALIDATION SELECTION -> TEST EVALUATION)")
    print("=" * 76)

    # 5. Train each classifier and evaluate on VALIDATION set
    for name, clf in candidate_models.items():
        print(f"\n[Training] {name}...")
        try:
            clf.fit(X_train_scaled, y_train)
            trained_models[name] = clf
        except Exception as e:
            print(f"  [Warning] Failed to train {name}: {e}. Skipping...")
            continue

        # Validation evaluation
        y_val_pred = clf.predict(X_val_scaled)
        y_val_prob = clf.predict_proba(X_val_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_val_pred

        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, zero_division=0)
        val_rec = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        try:
            val_auc = roc_auc_score(y_val, y_val_prob)
        except Exception:
            val_auc = 0.5

        val_metrics[name] = {
            "val_accuracy": round(float(val_acc), 4),
            "val_precision": round(float(val_prec), 4),
            "val_recall": round(float(val_rec), 4),
            "val_f1": round(float(val_f1), 4),
            "val_roc_auc": round(float(val_auc), 4),
        }

        # Held-out Test evaluation
        y_test_pred = clf.predict(X_test_scaled)
        y_test_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_test_pred

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        try:
            test_auc = roc_auc_score(y_test, y_test_prob)
        except Exception:
            test_auc = 0.5
        cm = confusion_matrix(y_test, y_test_pred)

        test_metrics[name] = {
            "test_accuracy": round(float(test_acc), 4),
            "test_precision": round(float(test_prec), 4),
            "test_recall": round(float(test_rec), 4),
            "test_f1": round(float(test_f1), 4),
            "test_roc_auc": round(float(test_auc), 4),
            "confusion_matrix": cm.tolist(),
        }

        print(f"  Validation F1 : {val_f1:.4f} | Validation Acc : {val_acc:.4f}")
        print(f"  Test F1       : {test_f1:.4f} | Test Acc       : {test_acc:.4f}")

        results_records.append({
            "Model": name,
            "Val_F1": val_f1,
            "Val_Accuracy": val_acc,
            "Test_F1": test_f1,
            "Test_Accuracy": test_acc,
            "Test_Precision": test_prec,
            "Test_Recall": test_rec,
            "Test_ROC_AUC": test_auc,
            "Test_Confusion_Matrix": cm.ravel().tolist(),
        })

    if not trained_models:
        raise RuntimeError("No models could be successfully trained.")

    # 6. Model Selection: STRICTLY based on VALIDATION F1-score (NOT test-set performance)
    best_model_name = max(val_metrics.keys(), key=lambda m: val_metrics[m]["val_f1"])
    best_val_f1 = val_metrics[best_model_name]["val_f1"]
    best_model = trained_models[best_model_name]
    best_test_metrics = test_metrics[best_model_name]

    results_df = pd.DataFrame(results_records).sort_values(by="Val_F1", ascending=False)

    print("\n" + "=" * 76)
    print("BENCHMARK SUMMARY TABLE:")
    print(results_df[["Model", "Val_F1", "Val_Accuracy", "Test_F1", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_ROC_AUC"]].to_string(index=False))
    print("=" * 76)
    print(f"\n>>> Best Model Selected on Validation F1-Score: '{best_model_name}' (Validation F1 = {best_val_f1:.4f}) <<<")
    print(f"    Held-Out Test Set Evaluation for {best_model_name}:")
    print(f"    - Accuracy : {best_test_metrics['test_accuracy']:.4f}")
    print(f"    - Precision: {best_test_metrics['test_precision']:.4f}")
    print(f"    - Recall   : {best_test_metrics['test_recall']:.4f}")
    print(f"    - F1-Score : {best_test_metrics['test_f1']:.4f}")
    print(f"    - ROC-AUC  : {best_test_metrics['test_roc_auc']:.4f}")
    print(f"    - Confusion Matrix (TN, FP, FN, TP): {best_test_metrics['confusion_matrix']}")
    print("=" * 76)

    # 7. Save Artifacts Consistently
    # (a) Save best model
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump({
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": FEATURE_COLS,
        "val_metrics": val_metrics[best_model_name],
        "test_metrics": best_test_metrics,
    }, best_model_path)
    print(f"[Artifact] Saved best model to '{best_model_path}'.")

    # (b) Save all trained models so Streamlit dropdown can genuinely select any model
    all_models_path = os.path.join(output_dir, "all_models.joblib")
    joblib.dump({
        name: {
            "model": clf,
            "feature_names": FEATURE_COLS,
            "val_metrics": val_metrics[name],
            "test_metrics": test_metrics[name],
        }
        for name, clf in trained_models.items()
    }, all_models_path)
    print(f"[Artifact] Saved all {len(trained_models)} trained models to '{all_models_path}'.")

    # (c) Save comparison results CSV
    csv_path = os.path.join(output_dir, "model_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[Artifact] Saved benchmark comparison to '{csv_path}'.")

    # (d) Save evaluation metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "selected_model": best_model_name,
            "selection_criterion": "Validation F1-score",
            "validation_metrics": val_metrics,
            "held_out_test_metrics": test_metrics,
            "feature_names": FEATURE_COLS,
        }, f, indent=2)
    print(f"[Artifact] Saved evaluation metrics to '{metrics_path}'.")

    # (e) Fit & Save SHAP explainer on background training sample
    try:
        background_sample = X_train_scaled[:100]
        # Save background for explainers
        joblib.dump(background_sample, os.path.join(output_dir, "shap_background.joblib"))

        # Fit TreeExplainer if best model is tree-based
        if hasattr(best_model, "feature_importances_"):
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.LinearExplainer(best_model, background_sample)

        shap_path = os.path.join(output_dir, "shap_explainer.joblib")
        joblib.dump(explainer, shap_path)
        print(f"[Artifact] Saved SHAP explainer to '{shap_path}'.")
    except Exception as e:
        print(f"[SHAP] Background explainer note: {e}")

    print(f"\n[Success] Training, benchmarking, and artifact generation complete in '{output_dir}'.")
    return {
        "best_model_name": best_model_name,
        "val_f1": best_val_f1,
        "test_metrics": best_test_metrics,
        "results_df": results_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Benchmark AI Image Detector Classifiers")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset images")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--max_samples", type=int, default=2000, help="Maximum balanced sample size for training")
    args = parser.parse_args()

    train_and_evaluate(args.data_dir, args.output_dir, args.max_samples)
