"""Electronic device detection via geometric heuristics."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..image_utils import ensure_color
from ..types import DetectionResult, Feature
from .base import FeatureDetector


class ElectronicsDetector(FeatureDetector):
    """Detect rectangular electronic displays like TVs and laptops."""

    def __init__(self, min_area_ratio: float = 0.05) -> None:
        super().__init__(Feature.ELECTRONICS)
        self.min_area_ratio = min_area_ratio

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        color_image = ensure_color(image)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = gray.shape[:2]
        image_area = float(h * w)
        best_confidence = 0.0
        best_box = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_ratio * image_area:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            box_area = cw * ch
            fill_ratio = area / max(1.0, box_area)
            aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
            if fill_ratio < 0.55:
                continue
            if aspect_ratio < 1.0 or aspect_ratio > 6.0:
                continue

            confidence = self._clip_confidence(min(0.95, (area / image_area) * 4))
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = (int(x), int(y), int(cw), int(ch))

        if best_box is None:
            return None

        return DetectionResult(
            feature=self.feature,
            confidence=best_confidence,
            details={"bounding_box": list(best_box)},
        )
