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
        
        # 即使没有匹配品牌，也尝试返回 OCR 识别的文本（低置信度）
        unmatched_ocr = self._get_all_ocr_texts(image)
        if unmatched_ocr:
            return unmatched_ocr
        
        return self._detect_via_templates(image)

    # ------------------------------------------------------------------
    # OCR-based detection
    # ------------------------------------------------------------------

    def _detect_via_text(self, image: np.ndarray) -> Optional[DetectionResult]:
        if not self.brand_keywords:
            return None

        try:
            text_regions = self._ocr.detect(image)
        except Exception as e:
            # OCR 检测失败，静默跳过
            return None
        
        if not text_regions:
            return None

        # 记录所有 OCR 识别的文本（用于调试）
        all_recognized_texts = [region.text for region in text_regions]
        if all_recognized_texts:
            print(f"    🔍 OCR 识别到 {len(all_recognized_texts)} 个文本区域: {', '.join(all_recognized_texts[:3])}{'...' if len(all_recognized_texts) > 3 else ''}")
        
        # 调试：显示品牌关键字（仅前3个）
        if self.brand_keywords:
            print(f"    🔑 匹配关键字: {', '.join([kw[1] for kw in self.brand_keywords[:3]])}...")

        for region in text_regions:
            normalized_text = self._normalize_label(region.text)
            if not normalized_text:
                continue
            
            # 尝试匹配品牌关键字
            for normalized_keyword, label in self.brand_keywords:
                if normalized_keyword and normalized_keyword in normalized_text:
                    x, y, w, h = region.bounding_box
                    print(f"    ✅ OCR 匹配成功: '{region.text}' 匹配品牌 '{label}'")
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
                # 反向匹配：检查品牌关键字是否在识别文本中
                # 要求：识别文本至少3个字符，且匹配长度 >= 4
                elif (normalized_keyword and 
                      len(normalized_text) >= 3 and 
                      len(normalized_keyword) >= 4 and 
                      normalized_text in normalized_keyword):
                    x, y, w, h = region.bounding_box
                    print(f"    ✅ OCR 部分匹配: '{region.text}' 部分匹配品牌 '{label}'")
                    return DetectionResult(
                        feature=self.feature,
                        confidence=self._clip_confidence(region.confidence * 0.8),  # 降低置信度
                        details={
                            "method": "ocr",
                            "brand": label,
                            "recognized_text": region.text,
                            "bounding_box": [int(x), int(y), int(w), int(h)],
                            "match_type": "partial"
                        },
                    )
        
        # OCR 识别了文本，但没有匹配品牌
        if all_recognized_texts:
            print(f"    ⚠️  OCR 识别了文本，但未匹配任何品牌关键字")
        return None
    
    def _get_all_ocr_texts(self, image: np.ndarray) -> Optional[DetectionResult]:
        """获取所有 OCR 识别的文本，即使没有匹配品牌也返回"""
        try:
            text_regions = self._ocr.detect(image)
        except Exception:
            return None
        
        if not text_regions:
            return None
        
        # 合并所有识别的文本
        all_texts = [region.text for region in text_regions if region.text.strip()]
        if not all_texts:
            return None
        
        # 计算平均置信度
        avg_confidence = sum(region.confidence for region in text_regions) / len(text_regions)
        
        # 返回低置信度的结果，包含所有识别的文本
        print(f"    📝 保存未匹配的 OCR 文本: {', '.join(all_texts[:3])}{'...' if len(all_texts) > 3 else ''}")
        return DetectionResult(
            feature=self.feature,
            confidence=self._clip_confidence(avg_confidence * 0.3),  # 低置信度
            details={
                "method": "ocr",
                "brand": None,
                "recognized_texts": all_texts,  # 所有识别的文本
                "match_type": "unmatched",
                "text_count": len(all_texts)
            },
        )

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
