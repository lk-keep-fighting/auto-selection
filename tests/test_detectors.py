"""Unit tests for individual detectors using synthetic images."""

from __future__ import annotations

import pytest

from local_product_recognition.detectors import (
    BrandLogoDetector,
    ChemicalDetector,
    ControlledPropDetector,
    ElectronicsDetector,
    PersonDetector,
    ToyDetector,
)
from local_product_recognition.types import Feature

from . import image_factory as factory


def _assert_detection(result, feature: Feature) -> None:
    assert result is not None, "Expected detection result"
    assert result.feature is feature
    assert 0.0 <= result.confidence <= 1.0


def test_person_detector_detects_skin_regions():
    detector = PersonDetector()
    image = factory.create_person_like_image()
    result = detector.detect(image)
    _assert_detection(result, Feature.PERSON)


def test_person_detector_ignores_blank_canvas():
    detector = PersonDetector()
    blank_image = factory.create_blank_image()
    assert detector.detect(blank_image) is None


@pytest.mark.parametrize("template", ["acme.png", "bolt.png"])
def test_logo_detector_detects_known_templates(template: str):
    detector = BrandLogoDetector(similarity_threshold=0.4)
    image = factory.create_logo_image(template)
    result = detector.detect(image)
    _assert_detection(result, Feature.BRAND_LOGO)


def test_logo_detector_returns_none_for_plain_image():
    detector = BrandLogoDetector()
    blank = factory.create_blank_image()
    assert detector.detect(blank) is None


def test_chemical_detector_detects_diamond():
    detector = ChemicalDetector()
    image = factory.create_chemical_image()
    result = detector.detect(image)
    _assert_detection(result, Feature.CHEMICAL)


def test_electronics_detector_detects_screen():
    detector = ElectronicsDetector()
    image = factory.create_electronics_image()
    result = detector.detect(image)
    _assert_detection(result, Feature.ELECTRONICS)


def test_controlled_prop_detector_detects_blade_shape():
    detector = ControlledPropDetector()
    image = factory.create_prop_image()
    result = detector.detect(image)
    _assert_detection(result, Feature.CONTROLLED_PROP)


def test_toy_detector_detects_bright_circle():
    detector = ToyDetector()
    image = factory.create_toy_image()
    result = detector.detect(image)
    _assert_detection(result, Feature.TOY)
