"""Toy detection leveraging bright colors and rounded shapes."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..image_utils import ensure_color
from ..types import DetectionResult, Feature
from .base import FeatureDetector


class ToyDetector(FeatureDetector):
    """Detects playful, toy-like regions using saturation and circularity."""

    def __init__(self, min_color_ratio: float = 0.015) -> None:
        super().__init__(Feature.TOY)
        self.min_color_ratio = min_color_ratio

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        color_image = ensure_color(image)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        colorful_mask = cv2.inRange(hsv, (10, 120, 120), (175, 255, 255))
        colorful_mask = cv2.medianBlur(colorful_mask, 5)

        h, w = color_image.shape[:2]
        ratio = cv2.countNonZero(colorful_mask) / max(1, h * w)
        if ratio < self.min_color_ratio:
            return None

        contours, _ = cv2.findContours(colorful_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circular_regions = 0
        max_circularity = 0.0
        image_area = float(h * w)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.004 * image_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.45:
                circular_regions += 1
                max_circularity = max(max_circularity, circularity)

        if circular_regions == 0 and ratio < self.min_color_ratio * 2.5:
            return None

        confidence = self._clip_confidence(min(0.9, ratio * 3 + max_circularity / 2))
        return DetectionResult(
            feature=self.feature,
            confidence=confidence,
            details={
                "color_ratio": ratio,
                "circular_regions": circular_regions,
            },
        )
