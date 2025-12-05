"""Helpers that generate synthetic images for detector tests."""

from __future__ import annotations

from importlib import resources
from typing import Tuple

import cv2
import numpy as np

from local_product_recognition.image_utils import ensure_color


Color = Tuple[int, int, int]


def create_blank_image(width: int = 320, height: int = 320, color: Color = (255, 255, 255)) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def create_person_like_image() -> np.ndarray:
    image = create_blank_image(320, 480)
    face_color = (180, 190, 210)
    body_color = face_color
    cv2.circle(image, (160, 150), 70, face_color, -1)
    cv2.rectangle(image, (120, 210), (200, 380), body_color, -1)
    cv2.rectangle(image, (80, 380), (120, 450), body_color, -1)
    cv2.rectangle(image, (200, 380), (240, 450), body_color, -1)
    return image


def create_logo_image(template_name: str = "acme.png") -> np.ndarray:
    template = _load_logo_template(template_name)
    canvas = create_blank_image(360, 360)
    h, w = template.shape[:2]
    top = 40
    left = 40
    canvas[top : top + h, left : left + w] = template
    return canvas


def create_text_logo_image(text: str = "ACME") -> np.ndarray:
    canvas = create_blank_image(480, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 5
    normalized_text = text.upper()
    target_width = int(canvas.shape[1] * 0.8)
    base_size, _ = cv2.getTextSize(normalized_text, font, 1.0, thickness)
    if base_size[0] == 0:
        font_scale = 1.0
    else:
        font_scale = target_width / base_size[0]
    font_scale = float(max(1.0, min(3.5, font_scale)))
    text_size, baseline = cv2.getTextSize(normalized_text, font, font_scale, thickness)
    origin_x = max(10, (canvas.shape[1] - text_size[0]) // 2)
    origin_y = max(text_size[1] + baseline, (canvas.shape[0] + text_size[1]) // 2)
    cv2.putText(
        canvas,
        normalized_text,
        (origin_x, origin_y),
        font,
        font_scale,
        (40, 40, 40),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return canvas


def create_chemical_image() -> np.ndarray:
    image = create_blank_image(320, 320)
    center = (160, 160)
    size = 100
    diamond = np.array([
        [center[0], center[1] - size // 2],
        [center[0] + size // 2, center[1]],
        [center[0], center[1] + size // 2],
        [center[0] - size // 2, center[1]],
    ])
    cv2.fillPoly(image, [diamond], (30, 30, 255))
    return image


def create_electronics_image() -> np.ndarray:
    image = create_blank_image(512, 320, color=(230, 230, 230))
    cv2.rectangle(image, (60, 60), (452, 260), (30, 30, 30), -1)
    cv2.rectangle(image, (60, 60), (452, 260), (80, 80, 80), 6)
    return image


def create_prop_image() -> np.ndarray:
    image = create_blank_image(320, 200, color=(245, 245, 245))
    cv2.rectangle(image, (40, 90), (280, 120), (90, 90, 90), -1)
    cv2.rectangle(image, (40, 90), (280, 120), (200, 200, 200), 4)
    return image


def create_toy_image() -> np.ndarray:
    image = create_blank_image(320, 320)
    cv2.circle(image, (160, 160), 80, (60, 20, 255), -1)
    cv2.circle(image, (120, 130), 18, (255, 255, 0), -1)
    cv2.circle(image, (200, 190), 15, (0, 255, 255), -1)
    return image


def _load_logo_template(name: str) -> np.ndarray:
    data_root = resources.files("local_product_recognition").joinpath("data", "logos", name)
    raw = np.frombuffer(data_root.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return ensure_color(image)
