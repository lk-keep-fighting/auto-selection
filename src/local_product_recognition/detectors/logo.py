"""Brand logo detection via OCR text matching with template fallback."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..image_utils import ensure_gray
from ..ocr import LightweightTextRecognizer
from ..types import DetectionResult, Feature
from .base import FeatureDetector

BrandKeyword = Tuple[str, str]  # (normalized_value, display_label)


@dataclass(frozen=True)
class _LogoTemplate:
    name: str
    brand: str
    image: np.ndarray


class BrandLogoDetector(FeatureDetector):
    """Detects brand logos either through OCR text matching or template search."""

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        brand_keywords: Optional[Sequence[str]] = None,
        text_recognizer: Optional[LightweightTextRecognizer] = None,
    ) -> None:
        super().__init__(Feature.BRAND_LOGO)
        self.similarity_threshold = similarity_threshold
        self._templates = self._load_templates()
        self.brand_keywords: List[BrandKeyword] = self._prepare_brand_keywords(
            brand_keywords, self._templates
        )
        self._ocr = text_recognizer or LightweightTextRecognizer()

    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        ocr_detection = self._detect_via_text(image)
        if ocr_detection is not None:
            return ocr_detection
        return self._detect_via_templates(image)

    # ------------------------------------------------------------------
    # OCR-based detection
    # ------------------------------------------------------------------

    def _detect_via_text(self, image: np.ndarray) -> Optional[DetectionResult]:
        if not self.brand_keywords:
            return None

        text_regions = self._ocr.detect(image)
        if not text_regions:
            return None

        for region in text_regions:
            normalized_text = self._normalize_label(region.text)
            if not normalized_text:
                continue
            for normalized_keyword, label in self.brand_keywords:
                if normalized_keyword and normalized_keyword in normalized_text:
                    x, y, w, h = region.bounding_box
                    return DetectionResult(
                        feature=self.feature,
                        confidence=self._clip_confidence(region.confidence),
                        details={
                            "method": "ocr",
                            "brand": label,
                            "recognized_text": region.text,
                            "bounding_box": [int(x), int(y), int(w), int(h)],
                        },
                    )
        return None

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _detect_via_templates(self, image: np.ndarray) -> Optional[DetectionResult]:
        if not self._templates:
            return None

        gray = ensure_gray(image)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        best_match: Optional[Tuple[_LogoTemplate, float, Tuple[int, int]]] = None

        for template in self._templates:
            th, tw = template.image.shape[:2]
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue

            result = cv2.matchTemplate(gray, template.image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < self.similarity_threshold:
                continue
            if best_match is None or max_val > best_match[1]:
                best_match = (template, float(max_val), max_loc)

        if best_match is None:
            return None

        template, score, location = best_match
        return DetectionResult(
            feature=self.feature,
            confidence=self._clip_confidence(score),
            details={
                "method": "template",
                "template": template.name,
                "brand": template.brand,
                "top_left": [int(location[0]), int(location[1])],
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            templates.append(
                _LogoTemplate(name=entry.name, brand=entry.stem.upper(), image=image)
            )
        return templates

    def _prepare_brand_keywords(
        self,
        provided: Optional[Sequence[str]],
        templates: Sequence[_LogoTemplate],
    ) -> List[BrandKeyword]:
        raw_keywords: List[str] = []
        if provided:
            raw_keywords.extend(provided)
        raw_keywords.extend(template.brand for template in templates)

        keywords: List[BrandKeyword] = []
        seen: set[str] = set()
        for keyword in raw_keywords:
            label = keyword.strip().upper()
            normalized = self._normalize_label(label)
            if not label or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            keywords.append((normalized, label))
        keywords.sort(key=lambda item: (-len(item[0]), item[0]))
        return keywords

    @staticmethod
    def _normalize_label(value: str) -> str:
        return "".join(ch for ch in value.upper() if ch.isalnum())
