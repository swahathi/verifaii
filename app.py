"""
Streamlit User Interface
AI-Generated Image Detection System using Metadata & Visual Features
"""

import streamlit as st
import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from predict_and_explain import AIDetectorInference

# Page Configuration
st.set_page_config(
    page_title="AI-Generated Image Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("AI-Generated Image Detection System")
st.markdown(
    "Data Mining & Machine Learning project analyzing **EXIF Metadata** and **Computer Vision Signals** "
    "(Noise Residual, Sharpness, Edge Density, Saturation, Entropy) to detect synthetic images."
)

# Sidebar
st.sidebar.header("System Controls")
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    ["XGBoost (Recommended)", "Random Forest", "Support Vector Machine", "Decision Tree", "Logistic Regression"],
)
st.sidebar.info(
    "**Data Mining Principle:**\n"
    "Missing EXIF metadata is **never** treated as definitive proof of AI generation. "
    "Visual frequency and sensor noise provide the primary discriminatory signals when metadata is absent."
)

# File Uploader
uploaded_file = st.file_uploader(
    "Upload an image for forensic analysis (JPEG, PNG, WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

# Initialize inference engine
detector = AIDetectorInference()

if uploaded_file is not None:
    # Save to temporary file for analysis
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            image = Image.open(tmp_path)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Run inference
        result = detector.predict(tmp_path)

        with col_pred:
            verdict = result["verdict"]
            confidence = result["confidence"]
            prob_ai = result["probability_ai"]
            prob_real = result["probability_real"]

            if verdict == "AI-Generated":
                st.error(f"### Prediction: AI-Generated ({confidence}% Confidence)")
            else:
                st.success(f"### Prediction: Real Photograph ({confidence}% Confidence)")

            st.write(f"**Model Probability:** {prob_ai * 100:.1f}% AI-Generated · {prob_real * 100:.1f}% Real")
            st.progress(prob_ai)

            # Plain-language explanation
            st.subheader("Explanation")
            st.write(result["explanation"])

            # Metadata status
            meta = result["metadata_raw"]
            if meta["has_exif"] == 1:
                st.info(f"**Metadata Status:** EXIF detected ({meta['raw_tag_count']} tags). Camera: {meta['camera_make'] or 'N/A'} {meta['camera_model'] or ''}")
            else:
                st.warning("**Metadata Status:** No EXIF headers found (common in web/social images). Classification based on visual noise & frequency.")

        # Features & Explainability Section
        st.markdown("---")
        st.subheader("Extracted Features & Explainability")

        col_feat, col_shap = st.columns([1, 1])

        with col_feat:
            st.write("**Extracted Feature Values:**")
            feat_df = pd.DataFrame([
                {"Feature": "Noise Residual (PRNU)", "Value": f"{result['features']['texture_noise_residual']:.4f}", "Benchmark": "0.28 - 0.55 (Real)"},
                {"Feature": "Color Saturation", "Value": f"{result['features']['color_saturation']:.4f}", "Benchmark": "0.20 - 0.40 (Real)"},
                {"Feature": "Laplacian Sharpness", "Value": f"{result['features']['sharpness']:.4f}", "Benchmark": "Optical falloff"},
                {"Feature": "Sobel Edge Density", "Value": f"{result['features']['edge_density'] * 100:.1f}%", "Benchmark": "10% - 22%"},
                {"Feature": "Shannon Entropy", "Value": f"{result['features']['image_entropy']:.2f} bits", "Benchmark": "7.0 - 7.9 bits"},
                {"Feature": "EXIF Camera Make", "Value": str(meta['camera_make'] or 'Missing'), "Benchmark": "Verified hardware"},
                {"Feature": "Software Tag", "Value": str(meta['software'] or 'None'), "Benchmark": "No AI generator"},
            ])
            st.dataframe(feat_df, use_container_width=True)

        with col_shap:
            st.write("**Feature Impact (SHAP Directional Push):**")
            fig, ax = plt.subplots(figsize=(6, 4))
            contribs = result["contributions"]
            names = [c["feature"] for c in contribs]
            shaps = [c["shap"] for c in contribs]
            colors = ["#ef4444" if s > 0 else "#10b981" for s in shaps]

            ax.barh(names, shaps, color=colors)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_xlabel("SHAP Value (Positive = AI, Negative = Real)")
            ax.set_title("Feature Contribution Breakdown")
            st.pyplot(fig)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("Please upload an image above or select one of the test samples to view detection results.")
