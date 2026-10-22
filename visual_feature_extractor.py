"""
Visual Feature Extraction Module
Extracts domain-relevant visual signals:
- Brightness & Contrast
- Laplacian Sharpness
- High-frequency noise residual (Sensor PRNU / Photon noise estimation)
- Sobel Edge Density
- Color Saturation & Chromatic Dispersion
- Shannon Grayscale Entropy
"""

import os
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy


def extract_visual_features(image_path_or_array, sample_size=(384, 384)) -> Dict[str, float]:
    """
    Extracts numerical visual features from an image.
    Supports file path or pre-loaded NumPy BGR image array.
    """
    features: Dict[str, float] = {
        "brightness": 0.5,
        "contrast": 0.5,
        "sharpness": 0.5,
        "edge_density": 0.2,
        "color_saturation": 0.4,
        "color_dispersion": 0.2,
        "texture_noise_residual": 0.25,
        "image_entropy": 7.0,
    }

    try:
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                return features
            # Load with OpenCV
            img = cv2.imread(image_path_or_array)
            if img is None:
                # Fallback to Pillow in case of non-standard format
                pil_img = Image.open(image_path_or_array).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            img = image_path_or_array

        if img is None or img.size == 0:
            return features

        # Resize for consistent, laptop-friendly feature computation
        if sample_size is not None:
            img = cv2.resize(img, sample_size, interpolation=cv2.INTER_AREA)

        # Convert color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Brightness & Contrast (Mean and standard deviation of luminance)
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        features["brightness"] = round(mean_lum / 255.0, 4)
        features["contrast"] = round(std_lum / 128.0, 4)

        # 2. Laplacian Sharpness (Variance of the Laplacian operator)
        # Genuine optical lenses exhibit distinctive focal plane falloff
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())
        # Normalized logarithmic scale [0.0, 1.0]
        features["sharpness"] = round(min(1.0, max(0.01, np.log10(lap_var + 1.0) / 3.5)), 4)

        # 3. Edge Density (Sobel gradient magnitude thresholding)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_threshold = 38.0
        edge_pixels = np.sum(sobel_mag > edge_threshold)
        features["edge_density"] = round(float(edge_pixels / gray.size), 4)

        # 4. Color Saturation & Color Channel Dispersion
        # Saturation channel in HSV (0-255)
        sat_channel = hsv[:, :, 1]
        features["color_saturation"] = round(float(np.mean(sat_channel)) / 255.0, 4)

        # Channel standard deviation dispersion
        b_mean, g_mean, r_mean = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
        channel_mean = (b_mean + g_mean + r_mean) / 3.0
        dispersion = np.sqrt(((b_mean - channel_mean)**2 + (g_mean - channel_mean)**2 + (r_mean - channel_mean)**2) / 3.0)
        features["color_dispersion"] = round(float(dispersion / 128.0), 4)

        # 5. Texture & High-Frequency Noise Residual (Sensor Noise Estimation)
        # Real camera CMOS sensors have PRNU photon noise; AI diffusion models exhibit micro-smoothing
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        noise_residual = gray.astype(np.float32) - blurred.astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(noise_residual**2)))
        features["texture_noise_residual"] = round(float(noise_rms / 32.0), 4)

        # 6. Shannon Grayscale Entropy (Information Density)
        entropy_val = float(shannon_entropy(gray))
        features["image_entropy"] = round(entropy_val, 4)

    except Exception as e:
        # Graceful fallback on malformed or corrupted inputs
        pass

    return features
