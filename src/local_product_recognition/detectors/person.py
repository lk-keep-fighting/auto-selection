"""Person detection using lightweight color heuristics."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..image_utils import ensure_color
from ..types import DetectionResult, Feature
from .base import FeatureDetector


class PersonDetector(FeatureDetector):
    """Detect human presence by identifying skin-like regions."""

    def __init__(self, min_skin_ratio: float = 0.01) -> None:
        super().__init__(Feature.PERSON)
        self.min_skin_ratio = min_skin_ratio
        self._kernel = np.ones((5, 5), np.uint8)

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        color_image = ensure_color(image)
        skin_mask = self._skin_mask(color_image)
        total_pixels = color_image.shape[0] * color_image.shape[1]
        skin_ratio = float(cv2.countNonZero(skin_mask) / max(1, total_pixels))
        if skin_ratio < self.min_skin_ratio:
            return None

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = color_image.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.005 * h * w:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = ch / max(1, cw)
            if aspect_ratio < 1.2:
                continue
            confidence = self._clip_confidence(min(0.99, skin_ratio * (area / (h * w)) * 12))
            return DetectionResult(
                feature=self.feature,
                confidence=confidence,
                details={
                    "skin_ratio": skin_ratio,
                    "bounding_box": [int(x), int(y), int(cw), int(ch)],
                },
            )
        return None

    def _skin_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 20, 60], dtype=np.uint8)
        upper1 = np.array([25, 180, 255], dtype=np.uint8)
        lower2 = np.array([160, 20, 60], dtype=np.uint8)
        upper2 = np.array([179, 180, 255], dtype=np.uint8)
        mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_ycc = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycc = np.array([255, 190, 150], dtype=np.uint8)
        mask_ycc = cv2.inRange(ycrcb, lower_ycc, upper_ycc)

        mask = cv2.bitwise_or(mask_hsv, mask_ycc)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return mask
