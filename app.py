"""
Streamlit User Interface
Verifaii — AI-Generated Image Detection System
Analyzes EXIF Metadata and Handcrafted Visual Features (Noise Residual, Sharpness, Edge Density, Saturation, Entropy)
"""

import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict_and_explain import AIDetectorInference, get_available_models


# Page Configuration
st.set_page_config(
    page_title="Verifaii — AI-Generated Image Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("Verifaii — AI-Generated Image Detection System")
st.markdown(
    "Data Mining prototype analyzing **EXIF Metadata** and **Handcrafted Visual Features** "
    "(Sensor Noise Residual, Laplacian Sharpness, Sobel Edge Density, Chromatic Saturation, Shannon Entropy) "
    "to distinguish real camera photographs from AI-generated imagery."
)

# Sidebar: Model Selection
st.sidebar.header("System Controls")
available_models = get_available_models("models")

if available_models:
    model_choice = st.sidebar.selectbox(
        "Select Trained Classifier",
        available_models,
        index=0,
        help="Choose between the automatically selected best model or benchmarked candidate classifiers.",
    )
    detector = AIDetectorInference(models_dir="models", selected_model_name=model_choice)
else:
    model_choice = "None"
    st.sidebar.selectbox(
        "Select Trained Classifier",
        ["No trained models found"],
        disabled=True,
    )
    detector = AIDetectorInference(models_dir="models")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Data Mining Principles:**\n"
    "- Missing EXIF metadata is **never** treated as standalone proof of AI generation.\n"
    "- Camera hardware tags can be stripped or spoofed.\n"
    "- Classifier decisions reflect statistical probabilities over combined visual and structural signals."
)

# Model Availability Warning if not trained yet
if not detector.is_ready:
    st.warning(
        "⚠️ **No trained model artifacts found in `models/`**.\n\n"
        "To train and benchmark the classifiers on your dataset, run the following command in PowerShell:\n"
        "```powershell\n"
        "python train_models.py --data_dir data --output_dir models --max_samples 2000\n"
        "```\n"
        "Once completed, reload this page to analyze images with genuine trained models."
    )

# File Uploader
uploaded_file = st.file_uploader(
    "Upload an image for forensic analysis (JPEG, PNG, WebP, etc.)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    # Save to a temporary file safely on Windows
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if not file_ext:
        file_ext = ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            image = Image.open(tmp_path)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        if not detector.is_ready:
            with col_pred:
                st.error("Cannot perform inference: No trained model is currently available.")
                st.info("Please train the model first using `python train_models.py`.")
        else:
            with st.spinner("Analyzing image signals and computing model inference..."):
                result = detector.predict(tmp_path)

            if not result.get("success", False):
                with col_pred:
                    st.error(f"Analysis Error: {result.get('error', 'Unknown error occurred.')}")
            else:
                with col_pred:
                    verdict = result["verdict"]
                    confidence = result["confidence"]
                    prob_ai = result["probability_ai"]
                    prob_real = result["probability_real"]
                    active_model = result.get("model_name", "Classifier")

                    if verdict == "AI-Generated":
                        st.error(f"### Prediction: AI-Generated ({confidence}% Probability)")
                    else:
                        st.success(f"### Prediction: Real Photograph ({confidence}% Probability)")

                    st.caption(f"Evaluated by: **{active_model}**")
                    st.write(f"**Model Estimates:** {prob_ai * 100:.1f}% AI-Generated · {prob_real * 100:.1f}% Real")
                    st.progress(prob_ai)

                    # Plain-language explanation
                    st.subheader("Forensic Explanation")
                    st.write(result["explanation"])

                    # Metadata status
                    meta = result["metadata_raw"]
                    if meta.get("has_exif", 0) == 1:
                        st.info(
                            f"**Metadata Status:** EXIF detected ({meta.get('raw_tag_count', 0)} tags). "
                            f"Camera: {meta.get('camera_make') or 'N/A'} {meta.get('camera_model') or ''}"
                        )
                    else:
                        st.warning(
                            "**Metadata Status:** No EXIF headers found. "
                            "This is standard for social/web images; classification is weighted on visual features."
                        )

                # Features & Explainability Section
                st.markdown("---")
                st.subheader("Extracted Features & Model Attribution")

                col_feat, col_shap = st.columns([1, 1])

                with col_feat:
                    st.write("**Extracted Signal Values:**")
                    feat_dict = result["features"]
                    feat_df = pd.DataFrame([
                        {"Feature": "Sensor Noise Residual (PRNU)", "Value": f"{feat_dict['texture_noise_residual']:.4f}", "Domain Interpretation": "High in physical sensors; lower in diffusion smoothing"},
                        {"Feature": "Color Saturation", "Value": f"{feat_dict['color_saturation']:.4f}", "Domain Interpretation": "Often elevated in generative styling"},
                        {"Feature": "Laplacian Sharpness", "Value": f"{feat_dict['sharpness']:.4f}", "Domain Interpretation": "Optical depth-of-field vs synthetic edge sharpness"},
                        {"Feature": "Sobel Edge Density", "Value": f"{feat_dict['edge_density'] * 100:.2f}%", "Domain Interpretation": "High-frequency structural detail density"},
                        {"Feature": "Shannon Entropy", "Value": f"{feat_dict['image_entropy']:.2f} bits", "Domain Interpretation": "Information distribution in grayscale channel"},
                        {"Feature": "EXIF Camera Make", "Value": str(meta.get("camera_make") or "Missing"), "Domain Interpretation": "Hardware provenance tag"},
                        {"Feature": "Software Tag", "Value": str(meta.get("software") or "None"), "Domain Interpretation": "Known AI generator / editing suite match"},
                    ])
                    st.dataframe(feat_df, use_container_width=True)

                with col_shap:
                    st.write(f"**Feature Impact Breakdown ({result.get('explanation_type', 'Attribution')}):**")
                    contribs = result.get("contributions", [])
                    if contribs:
                        # Display top 6 features by absolute impact
                        top_contribs = contribs[:6]
                        names = [c["feature"] for c in top_contribs][::-1]
                        impacts = [c["impact"] for c in top_contribs][::-1]
                        colors = ["#ef4444" if imp > 0 else "#10b981" for imp in impacts]

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.barh(names, impacts, color=colors)
                        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
                        ax.set_xlabel("Impact (Positive = AI-Generated, Negative = Real)")
                        ax.set_title("Top Contributing Features")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Feature impact breakdown is unavailable for the selected model.")

    finally:
        # Safe Windows temp file cleanup
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
else:
    st.info("Upload an image above to analyze its metadata and visual features.")
