"""Base detector definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import DetectionResult, Feature


class FeatureDetector(ABC):
    """Abstract base class for a feature detector."""

    feature: Feature

    def __init__(self, feature: Feature) -> None:
        self.feature = feature

    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """Return a detection result if the feature is present in the image."""

    def _clip_confidence(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))
