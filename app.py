"""
Verifaii — Forensic AI-Generated Image Detection System
UI Redesign: Editorial Typography & Minimal Forensic Aesthetic
Inspired by bold, oversized typographic design studio sensibilities.
"""

import os
import json
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict_and_explain import AIDetectorInference, get_available_models


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VERIFAII — Image Forensics Lab",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# EDITORIAL STYLESHEET (WARM OFF-WHITE, BOLD SANS-SERIF, VIVID RED ACCENT)
# -----------------------------------------------------------------------------
EDITORIAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800;900&family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg-canvas: #E8E6DF;
  --bg-card: #DFDCD4;
  --bg-card-subtle: #E2DFD7;
  --accent-red: #C90008;
  --accent-red-hover: #A60006;
  --ink-primary: #111111;
  --ink-secondary: #3A3A3A;
  --ink-muted: #666666;
  --border-dark: #111111;
  --border-light: rgba(17, 17, 17, 0.16);
  --font-editorial: 'Space Grotesk', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-body: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', Consolas, Monaco, "Courier New", monospace;
}

/* Eliminate Streamlit's Default Rainbow Top Decoration */
[data-testid="stDecoration"] {
  display: none !important;
}

/* Base Canvas */
html, body, [data-testid="stAppViewContainer"], .stApp {
  background-color: var(--bg-canvas) !important;
  color: var(--ink-primary) !important;
  font-family: var(--font-body) !important;
  -webkit-font-smoothing: antialiased;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 4rem !important;
  max-width: 1280px !important;
}

/* Sidebar Editorial Skin */
[data-testid="stSidebar"] {
  background-color: #DFDCD4 !important;
  border-right: 1.5px solid var(--border-dark) !important;
}

[data-testid="stSidebar"] * {
  color: var(--ink-primary) !important;
}

[data-testid="stSidebarCollapseButton"] button {
  color: var(--ink-primary) !important;
}

[data-testid="stSidebar"] .stSelectbox label {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--ink-muted) !important;
}

/* Streamlit Selectbox Overrides */
div[data-baseweb="select"] > div {
  background-color: var(--bg-canvas) !important;
  border: 1.5px solid var(--border-dark) !important;
  border-radius: 0px !important;
  color: var(--ink-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}

/* Top Navigation Bar */
.editorial-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  padding-bottom: 1.25rem;
  border-bottom: 1.5px solid var(--border-dark);
  margin-bottom: 2.75rem;
  width: 100%;
}

.nav-brand-group {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.nav-brand-title {
  font-family: var(--font-editorial);
  font-weight: 900;
  font-size: 1.4rem;
  letter-spacing: -0.04em;
  color: var(--ink-primary);
  text-transform: uppercase;
}

.nav-brand-tag {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--accent-red);
  text-transform: uppercase;
  border: 1px solid var(--accent-red);
  padding: 0.15rem 0.5rem;
}

.nav-status-group {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-family: var(--font-mono);
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-primary);
}

.status-dot-pulse {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #108548;
  box-shadow: 0 0 0 2px rgba(16, 133, 72, 0.25);
  display: inline-block;
}

.status-dot-pulse.warning {
  background-color: var(--accent-red);
  box-shadow: 0 0 0 2px rgba(201, 0, 8, 0.25);
}

/* Hero Section */
.hero-editorial-wrap {
  margin-bottom: 3.25rem;
}

.hero-kicker {
  font-family: var(--font-mono);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 1rem;
}

.hero-headline-oversized {
  font-family: var(--font-editorial);
  font-size: clamp(3.6rem, 11vw, 8.5rem);
  font-weight: 900;
  line-height: 0.88;
  letter-spacing: -0.055em;
  text-transform: uppercase;
  color: var(--accent-red);
  margin: 0 0 1.5rem 0;
  padding: 0;
  word-break: break-word;
  overflow-wrap: break-word;
}

.hero-subtitle-restrained {
  font-family: var(--font-body);
  font-size: clamp(1.05rem, 2vw, 1.35rem);
  font-weight: 500;
  color: var(--ink-primary);
  max-width: 680px;
  line-height: 1.45;
  margin: 0;
}

/* File Uploader Custom Styling */
[data-testid="stFileUploader"] {
  margin-top: 1rem;
  margin-bottom: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
  background-color: var(--bg-card) !important;
  border: 1.5px solid var(--border-dark) !important;
  border-radius: 0px !important;
  padding: 2.25rem 1.5rem !important;
  text-align: center !important;
  transition: all 0.2s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent-red) !important;
  background-color: #D6D3CB !important;
}

[data-testid="stFileUploaderDropzone"] button {
  background-color: var(--accent-red) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 0px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 1.4rem !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
  background-color: var(--accent-red-hover) !important;
}

/* Uploaded file badge */
[data-testid="stFileUploaderFile"] {
  background-color: var(--bg-card-subtle) !important;
  border: 1px solid var(--border-dark) !important;
  border-radius: 0px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
}

/* Native Buttons */
.stButton > button {
  border-radius: 0px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.76rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: 1.5px solid var(--border-dark) !important;
  background-color: transparent !important;
  color: var(--ink-primary) !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.15s ease !important;
}

.stButton > button:hover {
  background-color: var(--accent-red) !important;
  border-color: var(--accent-red) !important;
  color: #FFFFFF !important;
}

/* Specimen Frame */
.specimen-frame {
  border: 1.5px solid var(--border-dark);
  background-color: var(--bg-card-subtle);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}

.specimen-header-strip {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--border-light);
  padding-bottom: 0.5rem;
  margin-bottom: 0.75rem;
  display: flex;
  justify-content: space-between;
}

.specimen-meta-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem 1rem;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--ink-secondary);
}

.specimen-meta-item strong {
  color: var(--ink-primary);
  font-weight: 600;
}

/* Verdict Panels */
.verdict-editorial-panel {
  padding: 2.25rem 2rem;
  margin-bottom: 1.75rem;
}

.verdict-editorial-panel.ai-state {
  border: 2.5px solid var(--accent-red);
  background-color: #EDEAE1;
}

.verdict-editorial-panel.real-state {
  border: 2.5px solid var(--border-dark);
  background-color: #E2DFD7;
}

.verdict-kicker {
  font-family: var(--font-mono);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}

.ai-state .verdict-kicker {
  color: var(--accent-red);
}

.real-state .verdict-kicker {
  color: var(--ink-secondary);
}

.verdict-main-headline {
  font-family: var(--font-editorial);
  font-size: clamp(2.6rem, 5.5vw, 4.8rem);
  font-weight: 900;
  line-height: 0.9;
  letter-spacing: -0.045em;
  text-transform: uppercase;
  margin: 0 0 1.25rem 0;
  word-break: break-word;
}

.ai-state .verdict-main-headline {
  color: var(--accent-red);
}

.real-state .verdict-main-headline {
  color: var(--ink-primary);
}

/* Probability Ratio Meter */
.editorial-meter-block {
  margin: 1.5rem 0 0.5rem 0;
}

.editorial-meter-labels {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  margin-bottom: 0.5rem;
}

.meter-lbl-ai {
  color: var(--accent-red);
}

.meter-lbl-real {
  color: var(--ink-primary);
}

.editorial-meter-bar {
  width: 100%;
  height: 14px;
  background-color: var(--ink-primary);
  display: flex;
  overflow: hidden;
  border: 1px solid var(--border-dark);
}

.editorial-meter-ai-fill {
  background-color: var(--accent-red);
  height: 100%;
}

.editorial-meter-real-fill {
  background-color: var(--ink-primary);
  height: 100%;
}

/* Explanation Editorial Callout */
.forensic-callout {
  padding: 1.25rem 1.5rem;
  background-color: var(--bg-card-subtle);
  margin: 1.5rem 0;
  font-family: var(--font-body);
  font-size: 1.02rem;
  line-height: 1.55;
  color: var(--ink-primary);
}

.forensic-callout.ai-accent {
  border-left: 3.5px solid var(--accent-red);
}

.forensic-callout.neutral-accent {
  border-left: 3.5px solid var(--border-dark);
}

.forensic-callout strong {
  font-weight: 700;
}

/* Metadata Forensic Badges */
.meta-audit-card {
  border: 1px solid var(--border-dark);
  background-color: var(--bg-card-subtle);
  padding: 1rem 1.25rem;
  margin: 1rem 0 1.5rem 0;
}

.meta-audit-header {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--border-light);
  padding-bottom: 0.4rem;
  margin-bottom: 0.75rem;
}

.meta-audit-content {
  font-family: var(--font-mono);
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--ink-primary);
}

/* Editorial Section Dividers */
.editorial-section-hdr {
  display: flex;
  align-items: baseline;
  gap: 0.85rem;
  border-bottom: 1.5px solid var(--border-dark);
  padding-bottom: 0.75rem;
  margin-top: 3.5rem;
  margin-bottom: 2rem;
}

.editorial-section-num {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--accent-red);
}

.editorial-section-title {
  font-family: var(--font-editorial);
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  text-transform: uppercase;
  color: var(--ink-primary);
  margin: 0;
}

/* Editorial Data Table */
.forensic-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-body);
}

.forensic-table th {
  text-align: left;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-muted);
  border-bottom: 1.5px solid var(--border-dark);
  padding: 0.75rem 0.5rem;
}

.forensic-table td {
  padding: 0.75rem 0.5rem;
  border-bottom: 1px solid var(--border-light);
  font-size: 0.86rem;
  color: var(--ink-primary);
  vertical-align: middle;
}

.forensic-table tr:last-child td {
  border-bottom: 1.5px solid var(--border-dark);
}

.forensic-table .td-val {
  font-family: var(--font-mono);
  font-weight: 700;
  color: var(--accent-red);
}

.forensic-table .td-val.neutral {
  color: var(--ink-primary);
}

/* Sidebar Editorial Elements */
.sidebar-masthead {
  padding-bottom: 1.25rem;
  border-bottom: 1.5px solid var(--border-dark);
  margin-bottom: 1.5rem;
}

.sidebar-title {
  font-family: var(--font-editorial);
  font-size: 1.25rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  text-transform: uppercase;
  color: var(--ink-primary);
}

.sidebar-subtitle {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent-red);
  margin-top: 0.25rem;
}

.sidebar-metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem;
  margin-top: 0.75rem;
  margin-bottom: 1.5rem;
}

.sidebar-metric-tile {
  background-color: var(--bg-canvas);
  border: 1px solid var(--border-dark);
  padding: 0.6rem 0.75rem;
}

.sidebar-metric-val {
  font-family: var(--font-mono);
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--ink-primary);
}

.sidebar-metric-lbl {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-muted);
}

.sidebar-rule-box {
  background-color: var(--bg-canvas);
  border: 1px solid var(--border-dark);
  padding: 1rem;
  margin-top: 1rem;
  font-size: 0.82rem;
  line-height: 1.45;
}

.sidebar-rule-title {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-red);
  margin-bottom: 0.5rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .editorial-nav {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  .specimen-meta-grid {
    grid-template-columns: 1fr;
  }
  .verdict-editorial-panel {
    padding: 1.5rem 1.25rem;
  }
}
</style>
"""

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# BENCHMARK METRICS LOADER
# -----------------------------------------------------------------------------
def load_evaluation_metrics():
    metrics_path = os.path.join("models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


eval_metrics = load_evaluation_metrics()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & BENCHMARK AUDIT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-masthead">
          <div class="sidebar-title">VERIFAII</div>
          <div class="sidebar-subtitle">SYSTEM CONTROLS & BENCHMARKS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_models = get_available_models("models")

    if available_models:
        model_choice = st.selectbox(
            "Classifier Architecture",
            available_models,
            index=0,
            help="Select between the automatically chosen best model or candidate benchmark classifiers.",
        )
        detector = AIDetectorInference(models_dir="models", selected_model_name=model_choice)
    else:
        model_choice = "None"
        st.selectbox("Classifier Architecture", ["No trained models found"], disabled=True)
        detector = AIDetectorInference(models_dir="models")

    # Display Genuine Model Metrics if available in evaluation_metrics.json
    active_clean_name = detector.model_name
    if eval_metrics and "validation_metrics" in eval_metrics:
        val_data = eval_metrics["validation_metrics"].get(active_clean_name, {})
        test_data = eval_metrics.get("held_out_test_metrics", {}).get(active_clean_name, {})

        if val_data:
            st.markdown(
                f"""
                <div style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-muted); margin-top: 1rem;">
                  BENCHMARK: {active_clean_name}
                </div>
                <div class="sidebar-metric-grid">
                  <div class="sidebar-metric-tile">
                    <div class="sidebar-metric-val">{val_data.get('val_f1', 0.0):.3f}</div>
                    <div class="sidebar-metric-lbl">Val F1-Score</div>
                  </div>
                  <div class="sidebar-metric-tile">
                    <div class="sidebar-metric-val">{val_data.get('val_accuracy', 0.0)*100:.1f}%</div>
                    <div class="sidebar-metric-lbl">Val Accuracy</div>
                  </div>
                  <div class="sidebar-metric-tile">
                    <div class="sidebar-metric-val">{val_data.get('val_roc_auc', 0.0):.3f}</div>
                    <div class="sidebar-metric-lbl">ROC-AUC</div>
                  </div>
                  <div class="sidebar-metric-tile">
                    <div class="sidebar-metric-val">{test_data.get('test_accuracy', 0.0)*100:.1f}%</div>
                    <div class="sidebar-metric-lbl">Test Accuracy</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="sidebar-rule-box">
          <div class="sidebar-rule-title">Forensic Principles</div>
          <p style="margin: 0 0 0.5rem 0;">• Missing EXIF is <strong>never</strong> treated as standalone proof of AI synthesis.</p>
          <p style="margin: 0 0 0.5rem 0;">• Hardware provenance tags are frequently stripped by social networks.</p>
          <p style="margin: 0;">• Decisions reflect multi-signal statistical consensus across sensor PRNU noise, saturation, and edge structure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# TOP NAVIGATION BAR (WITH GENUINE STATUS)
# -----------------------------------------------------------------------------
if detector.is_ready:
    status_dot_class = "status-dot-pulse"
    status_text = f"ENGINE READY · {detector.model_name.upper()} ACTIVE"
else:
    status_dot_class = "status-dot-pulse warning"
    status_text = "MODELS UNTRAINED · ARTIFACTS MISSING"

st.markdown(
    f"""
    <header class="editorial-nav">
      <div class="nav-brand-group">
        <span class="nav-brand-title">VERIFAII</span>
        <span class="nav-brand-tag">FORENSIC LAB</span>
      </div>
      <div class="nav-status-group">
        <span class="{status_dot_class}"></span>
        <span>{status_text}</span>
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HERO SECTION (BOLD, OVERSIZED RED EDITORIAL TYPOGRAPHY)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-editorial-wrap">
      <div class="hero-kicker">IMAGE FORENSICS SYSTEM &mdash; REF. SPEC-2026</div>
      <h1 class="hero-headline-oversized">
        REAL OR<br>GENERATED?
      </h1>
      <p class="hero-subtitle-restrained">
        An image forensics tool powered by metadata and visual feature analysis.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# UNINITIALIZED MODEL WARNING
# -----------------------------------------------------------------------------
if not detector.is_ready:
    st.markdown(
        """
        <div style="border: 2px solid var(--accent-red); background-color: #EDEAE1; padding: 1.5rem; margin-bottom: 2rem;">
          <div style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; color: var(--accent-red); text-transform: uppercase; margin-bottom: 0.5rem;">
            ⚠️ System Artifacts Missing
          </div>
          <p style="margin: 0 0 0.75rem 0; font-size: 0.95rem;">
            No trained models were located in <code>models/</code>. Run training to generate classifiers and explainers:
          </p>
          <code style="display: block; background: #111111; color: #E8E6DF; padding: 0.75rem; font-family: var(--font-mono); font-size: 0.82rem;">
            python train_models.py --data_dir data --output_dir models
          </code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# IMAGE INPUT SECTION (FILE UPLOADER + CURATED QUICK SPECIMEN TESTS)
# -----------------------------------------------------------------------------
if "selected_sample_path" not in st.session_state:
    st.session_state.selected_sample_path = None
if "selected_sample_name" not in st.session_state:
    st.session_state.selected_sample_name = None

uploaded_file = st.file_uploader(
    "Upload image specimen for forensic evaluation (JPEG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    key="image_file_uploader",
)

# Quick sample testing buttons for instantaneous inspection
sample_ai_path = os.path.join("data", "ai", "0001.jpg")
sample_real_path = os.path.join("data", "real", "0003.jpg")
has_samples = os.path.exists(sample_ai_path) and os.path.exists(sample_real_path)

if has_samples and uploaded_file is None:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: -0.5rem; margin-bottom: 0.75rem;">
          <span style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-muted);">
            Curated Test Specimens:
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_btn_ai, col_btn_real, col_btn_clear = st.columns([1.2, 1.2, 2.5])
    with col_btn_ai:
        if st.button("AI Diffusion Specimen"):
            st.session_state.selected_sample_path = sample_ai_path
            st.session_state.selected_sample_name = "AI_Diffusion_0001.jpg"
            st.rerun()
    with col_btn_real:
        if st.button("Real Camera Specimen"):
            st.session_state.selected_sample_path = sample_real_path
            st.session_state.selected_sample_name = "Camera_Photo_0001.jpg"
            st.rerun()
    with col_btn_clear:
        if st.session_state.selected_sample_path:
            if st.button("Clear Specimen Selection"):
                st.session_state.selected_sample_path = None
                st.session_state.selected_sample_name = None
                st.rerun()


# Determine effective image target
active_image_path = None
specimen_display_name = None
is_temp_file = False

if uploaded_file is not None:
    # Upload takes priority over sample
    file_ext = os.path.splitext(uploaded_file.name)[1].lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        active_image_path = tmp.name
    specimen_display_name = uploaded_file.name
    is_temp_file = True
elif st.session_state.selected_sample_path and os.path.exists(st.session_state.selected_sample_path):
    active_image_path = st.session_state.selected_sample_path
    specimen_display_name = st.session_state.selected_sample_name
    is_temp_file = False


# -----------------------------------------------------------------------------
# RESTYLED RESTRAINED EDITORIAL CHART HELPER
# -----------------------------------------------------------------------------
def render_editorial_chart(contributions):
    if not contributions:
        return None

    top_contribs = contributions[:7]
    names = [c["feature"] for c in top_contribs][::-1]
    impacts = [c["impact"] for c in top_contribs][::-1]
    colors = ["#C90008" if imp > 0 else "#111111" for imp in impacts]

    fig, ax = plt.subplots(figsize=(7, 4.0), facecolor="#E8E6DF")
    ax.set_facecolor("#E8E6DF")

    y_pos = list(range(len(names)))
    bars = ax.barh(y_pos, impacts, color=colors, height=0.55, edgecolor="none")

    # Minimal Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#111111")
    ax.spines["bottom"].set_linewidth(1.2)

    # Grid & Zero Axis Line
    ax.axvline(0, color="#111111", linewidth=1.2, linestyle="-")
    ax.grid(axis="x", linestyle=":", color="#C8C5BC", linewidth=0.8, alpha=0.8)

    # Label styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold", color="#111111")
    ax.set_xlabel(
        "ATTRIBUTION IMPACT  (← REAL INFLUENCE  |  AI-GENERATED INFLUENCE →)",
        fontsize=7.5,
        fontweight="bold",
        color="#111111",
        labelpad=10,
    )
    ax.tick_params(axis="x", colors="#111111", labelsize=8)
    ax.tick_params(axis="y", length=0)

    # Text annotations on bars
    for bar, imp in zip(bars, impacts):
        width = bar.get_width()
        x_text = width + (0.012 if width >= 0 else -0.012)
        ha = "left" if width >= 0 else "right"
        text_color = "#C90008" if width >= 0 else "#111111"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{imp:+.3f}",
            va="center",
            ha=ha,
            fontsize=8,
            fontweight="bold",
            color=text_color,
            family="monospace",
        )

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SPECIMEN ANALYSIS PIPELINE & RENDERING
# -----------------------------------------------------------------------------
if active_image_path:
    try:
        col_specimen, col_verdict = st.columns([1, 1.25], gap="large")

        # Load specimen image
        pil_image = Image.open(active_image_path)
        img_width, img_height = pil_image.size
        img_mode = pil_image.mode
        file_bytes = os.path.getsize(active_image_path)
        file_kb = file_bytes / 1024.0

        with col_specimen:
            st.markdown(
                f"""
                <div class="specimen-frame">
                  <div class="specimen-header-strip">
                    <span>SPECIMEN RECORD</span>
                    <span>REF: {specimen_display_name}</span>
                  </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(pil_image, use_container_width=True)

            st.markdown(
                f"""
                  <div class="specimen-meta-grid" style="margin-top: 1rem;">
                    <div class="specimen-meta-item">DIMENSIONS: <strong>{img_width} × {img_height} PX</strong></div>
                    <div class="specimen-meta-item">COLOR FORMAT: <strong>{img_mode}</strong></div>
                    <div class="specimen-meta-item">FILE SIZE: <strong>{file_kb:.1f} KB</strong></div>
                    <div class="specimen-meta-item">EVALUATED BY: <strong>{detector.model_name}</strong></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not detector.is_ready:
            with col_verdict:
                st.error("Cannot perform inference: No trained model artifact found in `models/`.")
        else:
            with st.spinner("Extracting forensic metadata & visual signals..."):
                result = detector.predict(active_image_path)

            if not result.get("success", False):
                with col_verdict:
                    st.error(f"Analysis Pipeline Error: {result.get('error', 'Unknown failure')}")
            else:
                verdict = result["verdict"]
                confidence = result["confidence"]
                prob_ai = result["probability_ai"]
                prob_real = result["probability_real"]
                active_model = result.get("model_name", "Classifier")
                meta = result["metadata_raw"]
                feat_dict = result["features"]
                contributions = result.get("contributions", [])
                explanation_type = result.get("explanation_type", "Attribution")

                is_ai = verdict == "AI-Generated"
                panel_class = "ai-state" if is_ai else "real-state"
                verdict_title = "AI-GENERATED." if is_ai else "REAL PHOTOGRAPH."
                callout_class = "ai-accent" if is_ai else "neutral-accent"
                kicker_color = "var(--accent-red)" if is_ai else "var(--ink-secondary)"

                with col_verdict:
                    # Verdict Card
                    st.markdown(
                        f"""
                        <div class="verdict-editorial-panel {panel_class}">
                          <div class="verdict-kicker">FORENSIC VERDICT &bull; {active_model.upper()}</div>
                          <div class="verdict-main-headline">{verdict_title}</div>
                          <div style="font-family: var(--font-mono); font-size: 1.05rem; font-weight: 700; letter-spacing: 0.04em;">
                            {confidence}% CONFIDENCE ESTIMATE
                          </div>
                          
                          <!-- Dual-Segment Minimal Probability Meter -->
                          <div class="editorial-meter-block">
                            <div class="editorial-meter-labels">
                              <span class="meter-lbl-ai">AI-GENERATED: {prob_ai * 100:.1f}%</span>
                              <span class="meter-lbl-real">REAL PHOTO: {prob_real * 100:.1f}%</span>
                            </div>
                            <div class="editorial-meter-bar">
                              <div class="editorial-meter-ai-fill" style="width: {prob_ai * 100}%;"></div>
                              <div class="editorial-meter-real-fill" style="width: {prob_real * 100}%;"></div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Forensic Explanation Callout
                    st.markdown(
                        f"""
                        <div class="forensic-callout {callout_class}">
                          <div style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {kicker_color}; margin-bottom: 0.4rem;">
                            FORENSIC SYNTHESIS
                          </div>
                          {result["explanation"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Metadata Audit Box
                    has_exif = meta.get("has_exif", 0) == 1
                    if has_exif:
                        cam_str = f"{meta.get('camera_make') or 'Unknown'} {meta.get('camera_model') or ''}".strip()
                        soft_str = meta.get("software") or "None detected"
                        dt_str = meta.get("datetime_original") or "Unspecified"
                        meta_html = f"""
                        <div class="meta-audit-card">
                          <div class="meta-audit-header">EXIF METADATA AUDIT &mdash; PRESENT ({meta.get('raw_tag_count', 0)} TAGS)</div>
                          <div class="meta-audit-content">
                            <div>• HARDWARE: <strong>{cam_str}</strong></div>
                            <div>• SOFTWARE TAG: <strong>{soft_str}</strong></div>
                            <div>• TIMESTAMP: <strong>{dt_str}</strong></div>
                          </div>
                        </div>
                        """
                    else:
                        meta_html = """
                        <div class="meta-audit-card">
                          <div class="meta-audit-header">EXIF METADATA AUDIT &mdash; MISSING / STRIPPED</div>
                          <div class="meta-audit-content">
                            <div>• No camera hardware or capture headers detected.</div>
                            <div style="color: var(--ink-muted); font-size: 0.76rem; margin-top: 0.35rem;">
                              Standard for social platform compression. Classification relies on sensor noise residual and structural signals.
                            </div>
                          </div>
                        </div>
                        """
                    st.markdown(meta_html, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # SECTION 02: VISUAL SIGNAL EXTRACTION & EXPLAINABILITY
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="editorial-section-hdr">
              <span class="editorial-section-num">02</span>
              <h2 class="editorial-section-title">Visual Signal Extraction & Explainability</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_table, col_chart = st.columns([1.1, 1], gap="large")

        with col_table:
            st.markdown(
                """
                <div style="font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem; color: var(--ink-secondary);">
                  SIGNAL VALUES & DOMAIN INTERPRETATION
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Build Editorial Clean HTML Table
            cam_val = str(meta.get("camera_make") or "Missing")
            soft_val = str(meta.get("software") or "None")

            table_html = f"""
            <table class="forensic-table">
              <thead>
                <tr>
                  <th>Visual Signal Feature</th>
                  <th>Extracted Value</th>
                  <th>Forensic Interpretation</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Sensor Noise Residual (PRNU)</strong></td>
                  <td class="td-val">{feat_dict['texture_noise_residual']:.4f}</td>
                  <td>High in physical sensors; lower in diffusion smoothing</td>
                </tr>
                <tr>
                  <td><strong>Color Saturation</strong></td>
                  <td class="td-val">{feat_dict['color_saturation']:.4f}</td>
                  <td>Elevated saturation characteristic of synthetic rendering</td>
                </tr>
                <tr>
                  <td><strong>Laplacian Sharpness</strong></td>
                  <td class="td-val">{feat_dict['sharpness']:.4f}</td>
                  <td>Optical depth-of-field vs synthetic edge sharpness</td>
                </tr>
                <tr>
                  <td><strong>Sobel Edge Density</strong></td>
                  <td class="td-val">{feat_dict['edge_density'] * 100:.2f}%</td>
                  <td>Structural high-frequency boundary distribution</td>
                </tr>
                <tr>
                  <td><strong>Shannon Entropy</strong></td>
                  <td class="td-val">{feat_dict['image_entropy']:.2f} bits</td>
                  <td>Information distribution in luminance channels</td>
                </tr>
                <tr>
                  <td><strong>Mean Brightness</strong></td>
                  <td class="td-val neutral">{feat_dict['brightness']:.4f}</td>
                  <td>Global image luminance distribution</td>
                </tr>
                <tr>
                  <td><strong>Luminance Contrast</strong></td>
                  <td class="td-val neutral">{feat_dict['contrast']:.4f}</td>
                  <td>Dynamic range spread across pixel intensities</td>
                </tr>
                <tr>
                  <td><strong>Color Dispersion</strong></td>
                  <td class="td-val neutral">{feat_dict['color_dispersion']:.4f}</td>
                  <td>Inter-channel chromatic variation spread</td>
                </tr>
                <tr>
                  <td><strong>Camera Hardware Tag</strong></td>
                  <td class="td-val neutral">{cam_val}</td>
                  <td>Hardware provenance presence indicator</td>
                </tr>
                <tr>
                  <td><strong>Software Signature</strong></td>
                  <td class="td-val neutral">{soft_val}</td>
                  <td>Known AI generator or editing suite match</td>
                </tr>
              </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        with col_chart:
            st.markdown(
                f"""
                <div style="font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem; color: var(--ink-secondary);">
                  MODEL ATTRIBUTION BREAKDOWN ({explanation_type})
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = render_editorial_chart(contributions)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Feature attribution breakdown is not available for this classifier.")

    finally:
        # Safe cleanup of temp file if created
        if is_temp_file and active_image_path and os.path.exists(active_image_path):
            try:
                os.remove(active_image_path)
            except Exception:
                pass

else:
    # Empty State Guidance
    st.markdown(
        """
        <div style="border-top: 1.5px solid var(--border-dark); padding-top: 2rem; margin-top: 2rem;">
          <div style="font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 0.5rem;">
            AWAITING SPECIMEN INPUT
          </div>
          <p style="font-size: 1.05rem; color: var(--ink-primary); max-width: 640px; line-height: 1.5; margin: 0;">
            Upload an image above or select a curated specimen above to execute the forensic feature extraction and classification pipeline.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
