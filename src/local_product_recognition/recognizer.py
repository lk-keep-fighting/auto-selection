"""High level API that orchestrates all detectors."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .detectors import (
    BrandLogoDetector,
    ChemicalDetector,
    ControlledPropDetector,
    ElectronicsDetector,
    FeatureDetector,
    PersonDetector,
    ToyDetector,
)
from .image_utils import ImageInput, load_image
from .types import DetectionResult, Feature


class LocalProductImageRecognizer:
    """Runs every detector on an image and aggregates the results."""

    def __init__(self, detectors: Optional[Sequence[FeatureDetector]] = None) -> None:
        self.detectors: List[FeatureDetector] = list(detectors or self._default_detectors())

    def analyze(self, image_input: ImageInput) -> List[DetectionResult]:
        """Return structured detections for the provided image input."""

        image = load_image(image_input)
        results: List[DetectionResult] = []
        for detector in self.detectors:
            detection = detector.detect(image)
            if detection is not None:
                results.append(detection)
        return results

    def predict_labels(self, image_input: ImageInput) -> List[str]:
        """Convenience helper that only returns the feature labels."""

        return [result.feature.value for result in self.analyze(image_input)]

    def available_features(self) -> List[Feature]:
        """Expose which feature types are currently supported."""

        return [detector.feature for detector in self.detectors]

    @staticmethod
    def _default_detectors() -> Iterable[FeatureDetector]:
        return (
            PersonDetector(),
            BrandLogoDetector(),
            ChemicalDetector(),
            ElectronicsDetector(),
            ControlledPropDetector(),
            ToyDetector(),
        )
