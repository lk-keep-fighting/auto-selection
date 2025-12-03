"""Utility helpers for image loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image]


def load_image(image_input: ImageInput) -> np.ndarray:
    """Load an image input into an OpenCV-compatible BGR ndarray."""

    if isinstance(image_input, np.ndarray):
        image = image_input.copy()
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image from path: {path}")

    return ensure_color(image)


def ensure_color(image: np.ndarray) -> np.ndarray:
    """Ensure the ndarray is three-channel BGR."""

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale if necessary."""

    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
