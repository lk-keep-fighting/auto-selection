"""Detector for controlled or regulated props (e.g. blades)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..image_utils import ensure_color
from ..types import DetectionResult, Feature
from .base import FeatureDetector


class ControlledPropDetector(FeatureDetector):
    """Detects elongated metallic-looking shapes likely to be props or blades."""

    def __init__(self) -> None:
        super().__init__(Feature.CONTROLLED_PROP)

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        color_image = ensure_color(image)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = gray.shape[:2]
        image_area = float(h * w)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.003 * image_area:
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect
            if min(width, height) == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < 4.0:
                continue
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            fill_ratio = area / max(1.0, width * height)
            if fill_ratio < 0.35:
                continue
            confidence = self._clip_confidence(min(0.9, (area / image_area) * aspect_ratio / 5))
            return DetectionResult(
                feature=self.feature,
                confidence=confidence,
                details={
                    "aspect_ratio": aspect_ratio,
                    "angle": float(angle),
                    "box": box.tolist(),
                },
            )
        return None
