"""Integration style tests for the recognizer orchestration."""

from __future__ import annotations

import cv2
import pytest

from local_product_recognition import Feature, LocalProductImageRecognizer

from . import image_factory as factory


@pytest.fixture(scope="module")
def recognizer() -> LocalProductImageRecognizer:
    return LocalProductImageRecognizer()


@pytest.mark.parametrize(
    ("feature", "image_fn"),
    [
        (Feature.PERSON, factory.create_person_like_image),
        (Feature.BRAND_LOGO, factory.create_logo_image),
        (Feature.CHEMICAL, factory.create_chemical_image),
        (Feature.ELECTRONICS, factory.create_electronics_image),
        (Feature.CONTROLLED_PROP, factory.create_prop_image),
        (Feature.TOY, factory.create_toy_image),
    ],
)
def test_recognizer_identifies_each_feature(tmp_path, recognizer, feature, image_fn):
    image = image_fn()
    path = tmp_path / f"{feature.value}.png"
    cv2.imwrite(str(path), image)

    labels = recognizer.predict_labels(str(path))
    assert feature.value in labels


def test_recognizer_accepts_numpy_array(recognizer):
    image = factory.create_toy_image()
    results = recognizer.analyze(image)
    assert any(result.feature is Feature.TOY for result in results)


def test_blank_image_returns_no_features(recognizer):
    blank = factory.create_blank_image()
    assert recognizer.analyze(blank) == []


def test_available_features_matches_expected(recognizer):
    features = {feature for feature in recognizer.available_features()}
    assert features == {
        Feature.PERSON,
        Feature.BRAND_LOGO,
        Feature.CHEMICAL,
        Feature.ELECTRONICS,
        Feature.CONTROLLED_PROP,
        Feature.TOY,
    }
