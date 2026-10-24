"""
Prediction & Explainability Engine
Loads saved model and preprocessor to produce:
- Predicted class: Real Photograph (0) or AI-Generated (1)
- Model-estimated probability (calibrated confidence)
- Genuine feature contributions (using SHAP or model feature importance)
- Objective, transparent plain-language explanation without overclaiming
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from PIL import Image
import cv2

from metadata_extractor import extract_metadata
from visual_feature_extractor import extract_visual_features


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

FEATURE_DISPLAY_NAMES = {
    "has_exif": "EXIF Metadata Present",
    "has_camera_make": "Camera Manufacturer Tag",
    "has_camera_model": "Camera Model Tag",
    "has_datetime_original": "Capture Timestamp Tag",
    "has_software": "Editing Software Tag",
    "software_is_ai": "AI Generator Software Match",
    "brightness": "Mean Brightness",
    "contrast": "Luminance Contrast",
    "sharpness": "Laplacian Sharpness",
    "edge_density": "Sobel Edge Density",
    "color_saturation": "Color Saturation",
    "color_dispersion": "Color Dispersion",
    "texture_noise_residual": "Sensor Noise Residual",
    "image_entropy": "Shannon Entropy",
}


def get_available_models(models_dir: str = "models") -> List[str]:
    """
    Returns the list of available saved models in models_dir.
    """
    available = []
    best_path = os.path.join(models_dir, "best_model.joblib")
    all_path = os.path.join(models_dir, "all_models.joblib")

    if os.path.exists(best_path):
        try:
            art = joblib.load(best_path)
            best_name = art.get("model_name", "Best Model")
            available.append(f"Best Model ({best_name})")
        except Exception:
            available.append("Best Model")

    if os.path.exists(all_path):
        try:
            all_art = joblib.load(all_path)
            for m_name in all_art.keys():
                if m_name not in available:
                    available.append(m_name)
        except Exception:
            pass

    return available


class AIDetectorInference:
    def __init__(
        self,
        models_dir: str = "models",
        selected_model_name: Optional[str] = None
    ):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.all_models_path = os.path.join(models_dir, "all_models.joblib")
        self.explainer_path = os.path.join(models_dir, "shap_explainer.joblib")

        self.is_ready = False
        self.model = None
        self.model_name = "None"
        self.preprocessor = None
        self.explainer = None
        self.feature_names = FEATURE_COLS

        # Check if preprocessor and at least one model exists
        if os.path.exists(self.preprocessor_path) and (
            os.path.exists(self.model_path) or os.path.exists(self.all_models_path)
        ):
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)

                # Select requested model or default to best model
                if selected_model_name and os.path.exists(self.all_models_path):
                    all_models = joblib.load(self.all_models_path)
                    # Clean selection string
                    clean_name = selected_model_name
                    if clean_name.startswith("Best Model (") and clean_name.endswith(")"):
                        clean_name = clean_name[12:-1]

                    if clean_name in all_models:
                        self.model = all_models[clean_name]["model"]
                        self.model_name = clean_name
                        self.is_ready = True

                if not self.is_ready and os.path.exists(self.model_path):
                    best_art = joblib.load(self.model_path)
                    self.model = best_art["model"]
                    self.model_name = best_art.get("model_name", "Best Model")
                    self.is_ready = True

                # Attempt to load SHAP explainer if available
                if os.path.exists(self.explainer_path):
                    try:
                        self.explainer = joblib.load(self.explainer_path)
                    except Exception:
                        self.explainer = None
            except Exception as e:
                self.is_ready = False
                print(f"[Inference Init Error] {e}")

    def _compute_genuine_contributions(self, scaled_features: np.ndarray, raw_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Computes genuine feature contributions using SHAP TreeExplainer or trained model weights.
        Never fabricates arbitrary hardcoded numbers.
        """
        contributions = []
        explanation_type = "Model Feature Importance"

        # 1. Try SHAP first
        if self.explainer is not None:
            try:
                sv = self.explainer.shap_values(scaled_features)
                if isinstance(sv, list):
                    vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                elif len(getattr(sv, "shape", [])) == 3:
                    vals = sv[0, :, 1]
                elif len(getattr(sv, "shape", [])) == 2:
                    vals = sv[0]
                else:
                    vals = sv

                explanation_type = "SHAP (Marginal Contribution to AI Class)"
                for col_name, shap_val in zip(FEATURE_COLS, vals):
                    contributions.append({
                        "feature": FEATURE_DISPLAY_NAMES.get(col_name, col_name),
                        "raw_key": col_name,
                        "value": raw_features.get(col_name, 0.0),
                        "impact": round(float(shap_val), 4),
                        "explanation_type": explanation_type,
                    })
                contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
                return contributions
            except Exception:
                pass

        # 2. Try TreeExplainer on the current model directly
        if hasattr(self.model, "feature_importances_"):
            try:
                import shap
                tree_exp = shap.TreeExplainer(self.model)
                sv = tree_exp.shap_values(scaled_features)
                if isinstance(sv, list):
                    vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                elif len(getattr(sv, "shape", [])) == 3:
                    vals = sv[0, :, 1]
                elif len(getattr(sv, "shape", [])) == 2:
                    vals = sv[0]
                else:
                    vals = sv

                explanation_type = "SHAP (Marginal Contribution to AI Class)"
                for col_name, shap_val in zip(FEATURE_COLS, vals):
                    contributions.append({
                        "feature": FEATURE_DISPLAY_NAMES.get(col_name, col_name),
                        "raw_key": col_name,
                        "value": raw_features.get(col_name, 0.0),
                        "impact": round(float(shap_val), 4),
                        "explanation_type": explanation_type,
                    })
                contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
                return contributions
            except Exception:
                pass

        # 3. Model Weight / Coefficient Attribution for linear models
        if hasattr(self.model, "coef_"):
            try:
                coefs = self.model.coef_[0]
                # Linear contribution = weight * scaled_feature
                impacts = coefs * scaled_features[0]
                explanation_type = "Model Linear Weight Impact"
                for col_name, imp in zip(FEATURE_COLS, impacts):
                    contributions.append({
                        "feature": FEATURE_DISPLAY_NAMES.get(col_name, col_name),
                        "raw_key": col_name,
                        "value": raw_features.get(col_name, 0.0),
                        "impact": round(float(imp), 4),
                        "explanation_type": explanation_type,
                    })
                contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
                return contributions
            except Exception:
                pass

        # 4. Feature Importance * Deviation fallback
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            scaled_vec = scaled_features[0]
            # Directional impact = importance * sign of deviation
            explanation_type = "Feature Importance × Normalized Deviation"
            for col_name, imp, s_val in zip(FEATURE_COLS, importances, scaled_vec):
                dir_impact = imp * float(s_val)
                contributions.append({
                    "feature": FEATURE_DISPLAY_NAMES.get(col_name, col_name),
                    "raw_key": col_name,
                    "value": raw_features.get(col_name, 0.0),
                    "impact": round(float(dir_impact), 4),
                    "explanation_type": explanation_type,
                })
            contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
            return contributions

        # Default empty fallback if model provides no weights
        return contributions

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Runs complete inference and explainability pipeline.
        Handles unreadable images gracefully and never presents arbitrary fallback heuristics.
        """
        # Validate file existence
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found at '{image_path}'.",
                "is_ready": self.is_ready,
            }

        # Validate that image can be opened and decoded
        try:
            with Image.open(image_path) as test_img:
                test_img.verify()
        except Exception:
            # Try OpenCV read
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                return {
                    "success": False,
                    "error": "Image file is unreadable, corrupted, or in an unsupported format.",
                    "is_ready": self.is_ready,
                }

        # Extract domain features
        meta = extract_metadata(image_path)
        visual = extract_visual_features(image_path)

        feat_dict = {
            "has_exif": int(meta.get("has_exif", 0)),
            "has_camera_make": int(meta.get("has_camera_make", 0)),
            "has_camera_model": int(meta.get("has_camera_model", 0)),
            "has_datetime_original": int(meta.get("has_datetime_original", 0)),
            "has_software": int(meta.get("has_software", 0)),
            "software_is_ai": int(meta.get("software_is_ai", 0)),
            "brightness": float(visual.get("brightness", 0.5)),
            "contrast": float(visual.get("contrast", 0.5)),
            "sharpness": float(visual.get("sharpness", 0.5)),
            "edge_density": float(visual.get("edge_density", 0.2)),
            "color_saturation": float(visual.get("color_saturation", 0.4)),
            "color_dispersion": float(visual.get("color_dispersion", 0.2)),
            "texture_noise_residual": float(visual.get("texture_noise_residual", 0.25)),
            "image_entropy": float(visual.get("image_entropy", 7.0)),
        }

        # Check model availability: NEVER use silent fake heuristics
        if not self.is_ready or self.model is None or self.preprocessor is None:
            return {
                "success": False,
                "is_ready": False,
                "error": "No trained model available. Please run 'python train_models.py' first.",
                "features": feat_dict,
                "metadata_raw": meta,
                "visual_raw": visual,
            }

        # Format input feature vector with strictly identical ordering
        df_feat = pd.DataFrame([feat_dict])[FEATURE_COLS]
        scaled_feat = self.preprocessor.transform(df_feat)

        # Genuine Model Inference
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled_feat)[0]
            prob_ai = float(probs[1])
            prob_real = float(probs[0])
        else:
            pred = int(self.model.predict(scaled_feat)[0])
            prob_ai = 1.0 if pred == 1 else 0.0
            prob_real = 1.0 - prob_ai

        verdict = "AI-Generated" if prob_ai >= 0.5 else "Real Photograph"
        confidence = round((prob_ai if verdict == "AI-Generated" else prob_real) * 100.0, 1)

        # Compute genuine feature contributions
        contributions = self._compute_genuine_contributions(scaled_feat, feat_dict)
        explanation_type = contributions[0]["explanation_type"] if contributions else "Feature Attribution"

        # Transparent, probabilistic natural language explanation (no overclaiming)
        explanation_parts = []
        if verdict == "AI-Generated":
            explanation_parts.append(
                f"Classified as AI-Generated with an estimated probability of {prob_ai * 100:.1f}% by the {self.model_name} classifier."
            )
        else:
            explanation_parts.append(
                f"Classified as Real Photograph with an estimated probability of {prob_real * 100:.1f}% by the {self.model_name} classifier."
            )

        # Mention top genuine contributing signals
        if contributions:
            top_factors = contributions[:3]
            factor_descs = []
            for c in top_factors:
                dir_str = "AI-generated characteristics" if c["impact"] > 0 else "real photograph characteristics"
                factor_descs.append(f"{c['feature']} (value: {c['value']}, pushing toward {dir_str})")
            explanation_parts.append("Key contributing signals: " + "; ".join(factor_descs) + ".")

        # Metadata context without overclaiming
        if meta.get("has_exif", 0) == 1:
            make = meta.get("camera_make") or "Camera tag detected"
            model = meta.get("camera_model") or ""
            explanation_parts.append(f"EXIF metadata was present ({make} {model}). Note that metadata can be copied or stripped and is weighed alongside visual features.")
        else:
            explanation_parts.append("EXIF metadata was absent. This is standard across web and social media re-compression and is evaluated in context with visual frequency patterns.")

        explanation_parts.append("Note: Probabilities represent statistical model estimations on extracted features, not mathematical proof of origin.")

        return {
            "success": True,
            "is_ready": True,
            "model_name": self.model_name,
            "verdict": verdict,
            "confidence": confidence,
            "probability_ai": round(prob_ai, 4),
            "probability_real": round(prob_real, 4),
            "features": feat_dict,
            "metadata_raw": meta,
            "visual_raw": visual,
            "contributions": contributions,
            "explanation_type": explanation_type,
            "explanation": " ".join(explanation_parts),
        }
