"""
Metadata Extraction Module
Extracts EXIF metadata safely without crashing on corrupted headers.
Flags AI generator signatures while properly handling missing metadata.
"""

import os
from typing import Dict, Any, Optional
import exifread
from PIL import Image
from PIL.ExifTags import TAGS


AI_SOFTWARE_KEYWORDS = [
    "midjourney",
    "stable diffusion",
    "stablediffusion",
    "dall-e",
    "comfyui",
    "automatic1111",
    "novelai",
    "civitai",
    "invokeai",
    "flux",
    "generative fill",
    "firefly",
]


def extract_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts metadata features from an image.
    Never assumes missing EXIF implies AI generation.
    """
    metadata: Dict[str, Any] = {
        "has_exif": 0,
        "has_camera_make": 0,
        "has_camera_model": 0,
        "has_datetime_original": 0,
        "has_software": 0,
        "software_is_ai": 0,
        "camera_make": None,
        "camera_model": None,
        "datetime_original": None,
        "software": None,
        "iso_speed": -1.0,
        "focal_length": -1.0,
        "raw_tag_count": 0,
    }

    if not os.path.exists(image_path):
        return metadata

    # 1. First pass using ExifRead
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            if tags:
                metadata["has_exif"] = 1
                metadata["raw_tag_count"] = len(tags)

                if "Image Make" in tags:
                    make_str = str(tags["Image Make"]).strip()
                    if make_str:
                        metadata["camera_make"] = make_str
                        metadata["has_camera_make"] = 1

                if "Image Model" in tags:
                    model_str = str(tags["Image Model"]).strip()
                    if model_str:
                        metadata["camera_model"] = model_str
                        metadata["has_camera_model"] = 1

                if "EXIF DateTimeOriginal" in tags:
                    dt_str = str(tags["EXIF DateTimeOriginal"]).strip()
                    if dt_str:
                        metadata["datetime_original"] = dt_str
                        metadata["has_datetime_original"] = 1

                if "Image Software" in tags:
                    sw_str = str(tags["Image Software"]).strip()
                    if sw_str:
                        metadata["software"] = sw_str
                        metadata["has_software"] = 1
                        sw_lower = sw_str.lower()
                        if any(kw in sw_lower for kw in AI_SOFTWARE_KEYWORDS):
                            metadata["software_is_ai"] = 1

                if "EXIF ISOSpeedRatings" in tags:
                    try:
                        metadata["iso_speed"] = float(str(tags["EXIF ISOSpeedRatings"]))
                    except (ValueError, TypeError):
                        pass
    except Exception:
        # Graceful fallback for corrupted headers
        pass

    # 2. Secondary fallback pass with Pillow if ExifRead found no EXIF
    if metadata["has_exif"] == 0:
        try:
            with Image.open(image_path) as img:
                info = img._getexif()
                if info:
                    metadata["has_exif"] = 1
                    metadata["raw_tag_count"] = len(info)
                    for tag_id, value in info.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        if tag_name == "Make" and value:
                            metadata["camera_make"] = str(value).strip()
                            metadata["has_camera_make"] = 1
                        elif tag_name == "Model" and value:
                            metadata["camera_model"] = str(value).strip()
                            metadata["has_camera_model"] = 1
                        elif tag_name == "Software" and value:
                            sw_str = str(value).strip()
                            metadata["software"] = sw_str
                            metadata["has_software"] = 1
                            if any(kw in sw_str.lower() for kw in AI_SOFTWARE_KEYWORDS):
                                metadata["software_is_ai"] = 1
                        elif tag_name == "DateTimeOriginal" and value:
                            metadata["datetime_original"] = str(value).strip()
                            metadata["has_datetime_original"] = 1
        except Exception:
            pass

    return metadata
