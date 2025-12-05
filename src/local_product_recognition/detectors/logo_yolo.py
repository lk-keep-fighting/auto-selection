"""Logo detection using local YOLO models."""

from __future__ import annotations

from typing import List, Optional
import os

import numpy as np

from ..types import DetectionResult, Feature

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOLogoDetector:
    """
    Logo detector using local YOLO models.
    
    完全本地化的 Logo 检测方案，支持：
    1. 使用本地预训练模型
    2. 使用自己训练的 YOLOv8 模型
    3. 无需任何外部 API
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        device: str = "cpu"
    ):
        """
        Initialize local YOLO logo detector.
        
        Args:
            model_path: Path to local YOLO model weights (.pt file)
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            )
        
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            # Load custom local model
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f"✅ 已加载本地 Logo 检测模型: {model_path}")
            except Exception as e:
                print(f"❌ 加载模型失败: {str(e)}")
                self.model = None
        else:
            # No model provided or file doesn't exist
            if model_path:
                print(f"⚠️  模型文件不存在: {model_path}")
            print("\n" + "=" * 80)
            print("📝 Logo 检测需要本地模型文件")
            print("=" * 80)
            print("\n请选择以下方案之一:")
            print("\n方案 1: 训练自定义 Logo 模型（推荐 - 最灵活）")
            print("  1. 准备你需要检测的品牌 logo 图片和标注")
            print("  2. 使用 YOLOv8 训练:")
            print("     yolo train data=logo_data.yaml model=yolov8n.pt epochs=50")
            print("  3. 将训练好的模型路径设置为 LOGO_MODEL_PATH")
            
            print("\n方案 2: 下载开源 Logo 检测模型")
            print("  访问以下资源下载预训练模型:")
            print("  - GitHub: https://github.com/search?q=logo+detection+yolov8")
            print("  - Hugging Face: https://huggingface.co/models?search=logo+detection")
            
            print("\n方案 3: 使用现有模板匹配（已有功能）")
            print("  在 src/local_product_recognition/data/logos/ 添加 logo 图片")
            print("  使用传统 BrandLogoDetector（不需要深度学习模型）")
            print("\n" + "=" * 80)
            print("\n⚠️  Logo 检测功能已禁用，将跳过 Logo 检测\n")
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect logos in an image using local model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of logo detection results
        """
        if not self.model_loaded or self.model is None:
            return []  # 模型未加载，返回空列表
        
        try:
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
            
            # Process each detection
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Get logo brand name from model
                    brand_name = result.names.get(cls_id, f"logo_class_{cls_id}")
                    
                    detections.append(
                        DetectionResult(
                            feature=Feature.BRAND_LOGO,
                            confidence=confidence,
                            details={
                                "brand": brand_name,
                                "class_id": cls_id,
                                "bounding_box": [float(x) for x in bbox]
                            }
                        )
                    )
            
            return detections
            
        except Exception as e:
            print(f"⚠️  Logo 检测出错: {str(e)}")
            return []
    
    def get_highest_confidence_logo(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Get the logo detection with highest confidence.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Detection result with highest confidence, or None if no logos detected
        """
        detections = self.detect(image)
        
        if not detections:
            return None
        
        # Return detection with highest confidence
        return max(detections, key=lambda d: d.confidence)
