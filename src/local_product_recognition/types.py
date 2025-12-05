"""Common types used throughout the recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class Feature(str, Enum):
    """Semantic image features supported by the recognizer."""

    PERSON = "person"
    BRAND_LOGO = "brand_logo"
    CHEMICAL = "chemical"
    ELECTRONICS = "electronics"
    CONTROLLED_PROP = "controlled_prop"
    TOY = "toy"


@dataclass(frozen=True)
class DetectionResult:
    """Represents a single feature detection output."""

    feature: Feature
    confidence: float
    details: Dict[str, Any]
