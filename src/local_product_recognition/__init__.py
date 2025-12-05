"""Public exports for the local product recognition package."""

from .recognizer import LocalProductImageRecognizer
from .types import DetectionResult, Feature

__all__ = [
    "LocalProductImageRecognizer",
    "DetectionResult",
    "Feature",
]
