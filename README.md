# Local Product Image Recognition

This project implements a fully local image recognition pipeline that inspects
product images for sensitive visual features. The recognizer focuses on six
feature groups that are commonly required during e-commerce safety reviews:

1. Person (detecting visible people)
2. Brand logo marks
3. Chemical or hazardous material symbols
4. Electronic devices
5. Controlled or regulated props (e.g. blades)
6. Toys or child-focused items

The implementation relies solely on local classical computer vision algorithms
built on top of OpenCV. No external APIs or cloud inference services are used,
which makes the solution compliant with strict data residency requirements.

## Usage

```python
from local_product_recognition import LocalProductImageRecognizer

recognizer = LocalProductImageRecognizer()
results = recognizer.analyze("/path/to/image.jpg")

for detection in results:
    print(detection.feature.value, detection.confidence)
```

`analyze` accepts a filesystem path, Pillow image, or NumPy array and returns a
list of structured detections. A convenience `predict_labels` method is also
available when only the feature names are needed.

## Tests

```
pip install -e .[dev]
pytest
```

## Project Layout

```
src/local_product_recognition/
  detectors/       # Individual feature detectors
  data/logos/      # Lightweight synthetic logo templates used for logo matching
  image_utils.py   # Image loading helpers
  recognizer.py    # Public API surface
```

The detectors are intentionally modular so additional categories can be layered
on without modifying the public entrypoint.
