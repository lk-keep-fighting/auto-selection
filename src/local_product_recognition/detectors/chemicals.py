"""Chemical or hazardous material detector."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..image_utils import ensure_color
from ..types import DetectionResult, Feature
from .base import FeatureDetector


class ChemicalDetector(FeatureDetector):
    """Detects hazardous chemical labels such as orange/red diamonds."""

    def __init__(self, min_area_ratio: float = 0.01) -> None:
        super().__init__(Feature.CHEMICAL)
        self.min_area_ratio = min_area_ratio
        self._kernel = np.ones((7, 7), np.uint8)

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        color_image = ensure_color(image)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = self._hazard_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        h, w = color_image.shape[:2]
        image_area = float(h * w)
        mask_area_ratio = cv2.countNonZero(mask) / max(1.0, image_area)
        if mask_area_ratio < self.min_area_ratio:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_ratio * image_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) != 4:
                continue
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if min(width, height) == 0:
                continue
            square_ratio = max(width, height) / min(width, height)
            if not 0.8 <= square_ratio <= 1.35:
                continue
            confidence = self._clip_confidence(min(0.99, mask_area_ratio * 3.5))
            return DetectionResult(
                feature=self.feature,
                confidence=confidence,
                details={
                    "area_ratio": mask_area_ratio,
                    "vertices": len(approx),
                },
            )

        confidence = self._clip_confidence(mask_area_ratio)
        return DetectionResult(
            feature=self.feature,
            confidence=confidence,
            details={"area_ratio": mask_area_ratio},
        )

    def _hazard_mask(self, hsv: np.ndarray) -> np.ndarray:
        lower_orange = np.array([5, 80, 130], dtype=np.uint8)
        upper_orange = np.array([25, 255, 255], dtype=np.uint8)
        lower_red1 = np.array([0, 80, 120], dtype=np.uint8)
        upper_red1 = np.array([5, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 80, 120], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        return cv2.bitwise_or(mask_orange, cv2.bitwise_or(mask_red1, mask_red2))
