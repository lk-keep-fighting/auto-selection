"""Brand logo detection through template matching."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..image_utils import ensure_gray
from ..types import DetectionResult, Feature
from .base import FeatureDetector


@dataclass(frozen=True)
class _LogoTemplate:
    name: str
    image: np.ndarray


class BrandLogoDetector(FeatureDetector):
    """Detects known brand logos using lightweight template matching."""

    def __init__(self, similarity_threshold: float = 0.55) -> None:
        super().__init__(Feature.BRAND_LOGO)
        self.similarity_threshold = similarity_threshold
        self._templates = self._load_templates()

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        if not self._templates:
            return None

        gray = ensure_gray(image)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        best_match: Optional[Tuple[str, float, Tuple[int, int]]] = None

        for template in self._templates:
            th, tw = template.image.shape[:2]
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue

            result = cv2.matchTemplate(gray, template.image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < self.similarity_threshold:
                continue
            if best_match is None or max_val > best_match[1]:
                best_match = (template.name, float(max_val), max_loc)

        if best_match is None:
            return None

        template_name, score, location = best_match
        return DetectionResult(
            feature=self.feature,
            confidence=self._clip_confidence(score),
            details={
                "template": template_name,
                "top_left": [int(location[0]), int(location[1])],
            },
        )

    def _load_templates(self) -> List[_LogoTemplate]:
        templates: List[_LogoTemplate] = []
        try:
            data_root = resources.files("local_product_recognition").joinpath("data", "logos")
        except FileNotFoundError:
            return templates

        for entry in data_root.iterdir():
            if entry.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            raw = np.frombuffer(entry.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            templates.append(_LogoTemplate(name=entry.name, image=image))
        return templates
