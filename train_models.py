"""
Model Training & Benchmarking Pipeline
Trains and compares:
1. Random Forest Classifier
2. XGBoost Classifier
3. Support Vector Machine (SVM)
4. Decision Tree Classifier
5. Logistic Regression

Evaluates on unseen test split using Accuracy, Precision, Recall, F1, and Confusion Matrix.
Selects best model and saves to disk with Joblib.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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


def train_and_evaluate(data_dir: str, output_dir: str = "models", max_samples: int = 4000):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset & extract features
    records = scan_dataset_directory(data_dir)
    df = extract_features_from_dataset(records, max_samples=max_samples)

    # 2. Stratified train/val/test split to prevent leakage
    train_df, val_df, test_df = prepare_stratified_splits(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    # 3. Fit preprocessor ONLY on training set
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    # 4. Define the 5 classifiers
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1.5, probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    }

    results = []
    best_f1 = -1.0
    best_model_name = ""
    best_model = None

    print("\n" + "=" * 70)
    print("CLASSIFIER BENCHMARK COMPARISON (ON UNSEEN TEST SET)")
    print("=" * 70)

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Confusion Matrix (TN, FP, FN, TP)": cm.ravel().tolist(),
        })

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = clf

    # 5. Summary Table
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n" + "=" * 70)
    print("FINAL MODEL RANKING:")
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))
    print(f"\n--> Best Model Selected: {best_model_name} (F1: {best_f1:.4f})")
    print("=" * 70)

    # 6. Save best model artifact
    model_save_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump({"model_name": best_model_name, "model": best_model, "feature_names": FEATURE_COLS}, model_save_path)
    print(f"[Save] Saved best model to '{model_save_path}'.")

    # 7. Compute SHAP Feature Importance
    try:
        print("\nComputing SHAP feature importance...")
        explainer = shap.Explainer(best_model, X_train_scaled)
        shap_values = explainer(X_test_scaled[:200])
        joblib.dump(explainer, os.path.join(output_dir, "shap_explainer.joblib"))
        print("[Save] Saved SHAP explainer.")
    except Exception as e:
        print(f"[SHAP] Note: {e}")

    results_df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
    print(f"[Done] Training and evaluation complete! Results saved in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Image Detector Classifiers")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing real/ and ai/ images")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--max_samples", type=int, default=3000, help="Max sample images for training")
    args = parser.parse_args()

    train_and_evaluate(args.data_dir, args.output_dir, args.max_samples)
