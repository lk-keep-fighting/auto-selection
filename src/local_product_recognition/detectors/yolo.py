"""YOLO-based detection for various features."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..types import DetectionResult, Feature

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLODetector:
    """YOLO-based multi-feature detector using pre-trained models."""

    # COCO dataset class names that map to our features
    PERSON_CLASSES = {0}  # person
    ELECTRONICS_CLASSES = {63, 64, 65, 66, 67, 68, 72, 73, 76}  # laptop, mouse, remote, keyboard, cell phone, microwave, tv, book (tech), scissors
    TOY_CLASSES = {32, 33, 34, 35, 37, 38}  # sports ball, kite, baseball bat, baseball glove, skateboard, surfboard
    
    def __init__(
        self, 
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu"
    ) -> None:
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            )
        
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
    def detect_all(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run YOLO detection and return all detected features.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection results for all detected features
        """
        # Run YOLO inference
        results = self.model(
            image, 
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        detections = []
        
        # Track which features we've detected
        detected_features = {}
        
        # Process each detection
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Map COCO class to our features
                feature = None
                if cls_id in self.PERSON_CLASSES:
                    feature = Feature.PERSON
                elif cls_id in self.ELECTRONICS_CLASSES:
                    feature = Feature.ELECTRONICS
                elif cls_id in self.TOY_CLASSES:
                    feature = Feature.TOY
                
                if feature is not None:
                    # Keep only the highest confidence detection for each feature
                    if feature not in detected_features or confidence > detected_features[feature]["confidence"]:
                        bbox = box.xyxy[0].cpu().numpy()
                        detected_features[feature] = {
                            "confidence": confidence,
                            "class_id": cls_id,
                            "class_name": result.names[cls_id],
                            "bbox": [float(x) for x in bbox]
                        }
        
        # Convert to DetectionResult objects
        for feature, details in detected_features.items():
            detections.append(
                DetectionResult(
                    feature=feature,
                    confidence=details["confidence"],
                    details={
                        "yolo_class": details["class_name"],
                        "class_id": details["class_id"],
                        "bounding_box": details["bbox"]
                    }
                )
            )
        
        return detections
    
    def detect_specific(self, image: np.ndarray, feature: Feature) -> Optional[DetectionResult]:
        """
        Detect a specific feature type.
        
        Args:
            image: Input image as numpy array
            feature: Feature type to detect
            
        Returns:
            Detection result if feature is found, None otherwise
        """
        all_detections = self.detect_all(image)
        
        # Return the first detection matching the requested feature
        for detection in all_detections:
            if detection.feature == feature:
                return detection
        
        return None
