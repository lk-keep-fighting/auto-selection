"""Lightweight OCR helpers for brand logo text matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .image_utils import ensure_gray


@dataclass(frozen=True)
class RecognizedText:
    """Represents a single block of text recognized in an image."""

    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h


class LightweightTextRecognizer:
    """A tiny OCR helper tailored for simple, high-contrast product logos.

    The recognizer is purposely lightweight so it can run locally without
    external dependencies such as Tesseract. It works best for uppercase words
    rendered with relatively clean fonts (e.g. Hershey fonts used in OpenCV).
    """

    def __init__(
        self,
        char_set: Sequence[str] | str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        template_size: int = 32,
        min_word_score: float = 0.45,
        min_char_score: float = 0.35,
    ) -> None:
        if isinstance(char_set, str):
            iterable = list(dict.fromkeys(char_set))
        else:
            iterable = list(dict.fromkeys(char_set))
        self.char_set = [c.upper() for c in iterable]
        self.template_size = template_size
        self.min_word_score = min_word_score
        self.min_char_score = min_char_score
        self._templates = self._build_templates()

    def detect(self, image: np.ndarray) -> List[RecognizedText]:
        """Return text regions detected in the provided image."""

        if not self._templates:
            return []

        binary = self._prepare_binary(image)
        if binary is None:
            return []

        detections: List[RecognizedText] = []
        for x, y, w, h in self._find_text_regions(binary):
            region = binary[y : y + h, x : x + w]
            recognized = self._recognize_region(region)
            if recognized is None:
                continue
            text, score = recognized
            if score < self.min_word_score:
                continue
            detections.append(
                RecognizedText(
                    text=text,
                    confidence=float(score),
                    bounding_box=(int(x), int(y), int(w), int(h)),
                )
            )

        detections.sort(key=lambda det: (-det.confidence, det.bounding_box[1], det.bounding_box[0]))
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_binary(self, image: np.ndarray) -> np.ndarray | None:
        gray = ensure_gray(image)
        if gray is None:
            return None
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(cv2.countNonZero(thresh)) / float(thresh.size)
        if white_ratio > 0.5:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def _find_text_regions(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = binary.shape[:2]
        kernel_width = max(3, width // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 12 or h < 12:
                continue
            if w * h < 200:
                continue
            aspect = w / float(h)
            if aspect < 0.5 and w < 30:
                continue
            regions.append((x, y, w, h))

        regions.sort(key=lambda box: (box[1], box[0]))
        return regions[:32]

    def _recognize_region(self, region: np.ndarray) -> Tuple[str, float] | None:
        cropped = self._crop_to_content(region)
        if cropped is None:
            return None

        mask = (cropped > 0).astype(np.uint8)
        if mask.size == 0:
            return None

        segments = self._segment_characters(mask)
        if not segments:
            return None

        chars: List[str] = []
        scores: List[float] = []
        for segment in segments:
            prepared = self._prepare_character(segment)
            if prepared is None:
                continue
            char, score = self._match_character(prepared)
            if char is None or score < self.min_char_score:
                continue
            chars.append(char)
            scores.append(score)

        if not chars:
            return None

        text = "".join(chars)
        avg_score = float(sum(scores) / len(scores))
        return text, avg_score

    def _segment_characters(self, mask: np.ndarray) -> List[np.ndarray]:
        projection = mask.sum(axis=0)
        threshold = max(1, int(mask.shape[0] * 0.2))

        segments: List[np.ndarray] = []
        start = None
        for idx, value in enumerate(projection):
            if value > threshold:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    end = idx
                    if end - start >= 2:
                        left = max(0, start - 1)
                        right = min(mask.shape[1], end + 1)
                        segments.append(mask[:, left:right])
                    start = None
        if start is not None:
            end = mask.shape[1]
            if end - start >= 2:
                left = max(0, start - 1)
                segments.append(mask[:, left:end])
        return segments

    def _prepare_character(self, segment: np.ndarray) -> np.ndarray | None:
        cropped = self._crop_to_content((segment > 0).astype(np.uint8) * 255)
        if cropped is None:
            return None
        return self._resize_with_padding(cropped, self.template_size)

    def _match_character(self, char_img: np.ndarray) -> Tuple[str | None, float]:
        char_vector = (char_img / 255.0).astype(np.float32)
        if not np.any(char_vector):
            return None, 0.0

        best_char: str | None = None
        best_score = 0.0
        for char, variants in self._templates.items():
            for template in variants:
                score = self._cosine_similarity(char_vector, template)
                if score > best_score:
                    best_score = score
                    best_char = char
        return best_char, best_score

    def _build_templates(self) -> dict[str, List[np.ndarray]]:
        fonts = [
            (cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2),
            (cv2.FONT_HERSHEY_DUPLEX, 1.0, 2),
            (cv2.FONT_HERSHEY_PLAIN, 1.5, 2),
        ]
        templates: dict[str, List[np.ndarray]] = {}
        canvas_size = self.template_size * 2

        for raw_char in self.char_set:
            char = raw_char.upper()
            variants: List[np.ndarray] = []
            for font_face, scale, thickness in fonts:
                canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                baseline = canvas_size // 2 + int(scale * 4)
                cv2.putText(
                    canvas,
                    char,
                    (4, baseline),
                    font_face,
                    scale,
                    255,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cropped = self._crop_to_content(canvas)
                if cropped is None:
                    continue
                prepared = self._resize_with_padding(cropped, self.template_size)
                variants.append((prepared / 255.0).astype(np.float32))
            if variants:
                templates[char] = variants
        return templates

    @staticmethod
    def _resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        scale = min((target_size - 4) / h, (target_size - 4) / w)
        scale = max(scale, 0.1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        return canvas

    @staticmethod
    def _crop_to_content(image: np.ndarray) -> np.ndarray | None:
        if np.count_nonzero(image) == 0:
            return None
        coords = cv2.findNonZero(image)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_vec = a.flatten()
        b_vec = b.flatten()
        denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / denom)
