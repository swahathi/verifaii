# AI-Generated Image Detection System Using Metadata & Visual Feature Analysis

A Data Mining and Machine Learning project that classifies images as **Real** or **AI-Generated** using computer vision visual feature extraction and EXIF metadata analysis.

---

## 1. Project Overview & Methodology
This system bridges digital forensics, classical computer vision, and machine learning to distinguish camera-captured photographs from AI-generated synthesis (Stable Diffusion, Midjourney, DALL-E, etc.).

### A. Dual Feature Extraction
1. **Metadata Features (EXIF):**
   - Camera make and model presence
   - Shutter capture timestamp (`DateTimeOriginal`)
   - Software tags (including detection of AI tools: `Midjourney`, `ComfyUI`, `Automatic1111`, `Stable Diffusion`)
   - Total EXIF tag count and missingness indicators
   - *Crucial Rule:* Missing metadata is **never** treated as definitive proof of AI generation. Millions of authentic photos have EXIF stripped by social media (WhatsApp, Twitter, Instagram). Visual signals provide the authoritative classification basis.

2. **Visual Signal Features:**
   - **Brightness & Contrast:** Mean luminance and luminance standard deviation.
   - **Sharpness:** Laplacian operator variance $\text{Var}(\nabla^2 I)$.
   - **Texture & Sensor Noise Residual:** PRNU / photon shot noise estimation via high-pass filtering (real camera sensors exhibit natural Poisson-Gaussian noise; diffusion models create micro-smoothing or synthetic high-frequency residuals).
   - **Edge Density:** Sobel gradient magnitude thresholding (percentage of edge pixels).
   - **Color Saturation & Gamut:** Mean HSV saturation and channel variance (generative models often present hyper-vivid or biased gamuts).
   - **Information Entropy:** Shannon entropy $H = -\sum p_i \log_2(p_i)$ of grayscale distribution.

---

## 2. Directory Structure
```text
ai_image_detector/
├── data/
│   ├── real/                 # Real photographic images
│   └── ai/                   # AI-generated images
├── models/
│   ├── best_model.joblib     # Saved XGBoost / Random Forest model
│   └── preprocessor.joblib   # Saved scaler and feature imputer
├── dataset_loader.py         # Inspects dataset and creates stratified splits
├── metadata_extractor.py     # Extracts and parses EXIF metadata safely
├── visual_feature_extractor.py # OpenCV & skimage visual feature pipeline
├── train_models.py           # Trains & evaluates 5 classifiers (RF, XGB, SVM, DT, LR)
├── predict_and_explain.py    # Inference engine with SHAP explainability
├── app.py                    # Streamlit web application
├── experiment_notebook.ipynb # Jupyter notebook for interactive experimentation
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 3. Dataset Setup (Kaggle)
1. Download the dataset from Kaggle:
   **URL:** https://www.kaggle.com/datasets/malaxmels/ai-detection-dataset
2. Extract the downloaded archive into a `data/` folder in the project root:
   ```bash
   mkdir -p data
   # Extract real images to data/real/ and AI images to data/ai/
   ```
   *Note: `dataset_loader.py` automatically inspects whatever subfolder names exist (e.g. `real`/`fake`, `0`/`1`, `authentic`/`generated`).*

---

## 4. Local Installation & Execution

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Train & Evaluate Classifiers
Train all 5 classifiers, output the comparison table, confusion matrices, and save the best model:
```bash
python train_models.py --data_dir data/ --output_dir models/
```

### Step 3: Launch Streamlit Web UI
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser to test images, view SHAP waterfall explanations, and inspect feature values!


Deployed at : https://verifaii.streamlit.app/
