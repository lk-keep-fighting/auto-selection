#!/usr/bin/env python3
"""批量分析图片的主程序"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from local_product_recognition import LocalProductImageRecognizer
from local_product_recognition.types import DetectionResult
from local_product_recognition.detectors.yolo import YOLODetector
from local_product_recognition.detectors.logo_yolo import YOLOLogoDetector


def analyze_images_in_folder(
    folder_path: str, 
    use_yolo: bool = True,
    confidence_threshold: float = 0.5,
    enable_logo_detection: bool = False,
    logo_model_path: Optional[str] = None
) -> dict:
    """批量分析指定文件夹中的所有图片
    
    Args:
        folder_path: 图片文件夹路径
        use_yolo: 是否使用 YOLO 检测器（默认 True）
        confidence_threshold: 置信度阈值，只有大于此值的检测才被认为有效（默认 0.5）
        enable_logo_detection: 是否启用 Logo 检测（默认 False）
        logo_model_path: Logo 检测模型路径
        
    Returns:
        包含所有分析结果的字典
    """
    # 创建 YOLO 检测器或使用传统检测器
    if use_yolo:
        print(f"🤖 使用 YOLO 深度学习模型进行检测...")
        print(f"⚙️  置信度阈值: {confidence_threshold:.0%} (低于此值的检测将被归类为 passed)")
        logo_status = "开启" if enable_logo_detection else "关闭"
        print(f"🏷️  Logo 检测: {logo_status}\n")
        
        yolo_detector = YOLODetector(
            model_name="yolov8n.pt",  # 使用最小的模型，速度快
            confidence_threshold=0.25,  # YOLO 内部阈值设为较低值，在后处理中过滤
            device="cpu"  # 使用 CPU，如果有 GPU 可改为 "cuda"
        )
        
        # 初始化 Logo 检测器
        logo_detector = None
        if enable_logo_detection:
            logo_detector = YOLOLogoDetector(
                model_path=logo_model_path,
                confidence_threshold=0.25,
                device="cpu"
            )
        
        recognizer = None  # YOLO 检测器直接使用，不需要 recognizer
    else:
        print(f"📊 使用传统 OpenCV 算法进行检测...")
        print(f"⚙️  置信度阈值: {confidence_threshold:.0%} (低于此值的检测将被归类为 passed)\n")
        recognizer = LocalProductImageRecognizer()
        yolo_detector = None
        logo_detector = None
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # 获取文件夹中的所有图片文件
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return {}
    
    image_files = [
        f for f in folder.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"⚠️  文件夹中没有找到图片文件: {folder_path}")
        return {}
    
    # 按文件名排序
    image_files.sort()
    
    print(f"📁 找到 {len(image_files)} 张图片")
    print(f"🔍 开始分析...\n")
    print("=" * 80)
    
    # 分析结果汇总
    all_results = {}
    
    # 逐个分析图片
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 分析: {image_file.name}")
        
        try:
            # 分析图片
            if use_yolo:
                # 使用 YOLO 检测
                import cv2
                img = cv2.imread(str(image_file))
                results: List[DetectionResult] = yolo_detector.detect_all(img)
                
                # 如果启用了 Logo 检测，追加 Logo 检测结果
                if logo_detector is not None:
                    logo_results = logo_detector.detect(img)
                    if logo_results:
                        results.extend(logo_results)
            else:
                # 使用传统方法
                results: List[DetectionResult] = recognizer.analyze(str(image_file))
            
            # 格式化结果（保留所有检测结果，包括低置信度的）
            formatted_results = []
            valid_detections = 0  # 有效检测数（高于阈值）
            
            if results:
                for detection in results:
                    formatted_results.append({
                        "feature": detection.feature.value,
                        "confidence": round(detection.confidence, 4)
                    })
                    if detection.confidence >= confidence_threshold:
                        valid_detections += 1
                
                # 显示检测结果
                if valid_detections > 0:
                    print(f"  ✅ 检测到 {valid_detections} 个有效特征 (>={confidence_threshold:.0%}):")
                    for detection in results:
                        if detection.confidence >= confidence_threshold:
                            print(f"     - {detection.feature.value}: {detection.confidence:.2%}")
                    # 显示低置信度的检测
                    low_conf_count = len(results) - valid_detections
                    if low_conf_count > 0:
                        print(f"  ⚠️  {low_conf_count} 个低置信度检测 (<{confidence_threshold:.0%}):")
                        for detection in results:
                            if detection.confidence < confidence_threshold:
                                print(f"     - {detection.feature.value}: {detection.confidence:.2%}")
                else:
                    print(f"  ⚠️  检测到 {len(results)} 个特征，但都低于阈值 {confidence_threshold:.0%}:")
                    for detection in results:
                        print(f"     - {detection.feature.value}: {detection.confidence:.2%}")
            else:
                print(f"  ℹ️  未检测到任何特征")
            
            # 保存所有检测结果（包括低置信度的）
            all_results[image_file.name] = {
                "detections": formatted_results,
                "valid_count": valid_detections
            }
                
        except Exception as e:
            print(f"  ❌ 分析失败: {str(e)}")
            all_results[image_file.name] = {"error": str(e)}
    
    print("\n" + "=" * 80)
    print(f"\n✨ 分析完成! 共处理 {len(image_files)} 张图片\n")
    
    return all_results


def reorganize_results(results: dict, confidence_threshold: float = 0.5) -> dict:
    """重新整理结果，将通过和未通过的图片分类
    
    Args:
        results: 原始分析结果字典
        confidence_threshold: 置信度阈值
        
    Returns:
        重新整理后的结果字典，包含 passed 和 detected 两个数组
    """
    passed = []  # 没有有效特征的图片（所有检测都低于阈值或未检测到）
    detected = []  # 检测到有效特征的图片
    
    for image_name, data in results.items():
        if isinstance(data, dict) and "detections" in data:
            detections = data["detections"]
            valid_count = data.get("valid_count", 0)
            
            if valid_count > 0:
                # 有有效检测（高于阈值）
                # 分离有效和无效的检测
                valid_features = []
                low_confidence_features = []
                
                for detection in detections:
                    if detection.get("confidence", 0) >= confidence_threshold:
                        valid_features.append(detection)
                    else:
                        low_confidence_features.append(detection)
                
                detected.append({
                    "image": image_name,
                    "features": valid_features,
                    "low_confidence_detections": low_confidence_features if low_confidence_features else None
                })
            else:
                # 没有有效检测，归入 passed
                passed_item = {"image": image_name}
                
                # 如果有低置信度的检测，也保留下来
                if detections:
                    passed_item["low_confidence_detections"] = detections
                
                passed.append(passed_item)
        elif isinstance(data, dict) and "error" in data:
            # 处理错误情况
            detected.append({
                "image": image_name,
                "error": data["error"]
            })
    
    return {
        "passed": passed,
        "detected": detected,
        "summary": {
            "total": len(results),
            "passed_count": len(passed),
            "detected_count": len(detected),
            "confidence_threshold": confidence_threshold
        }
    }


def save_results_to_json(
    results: dict, 
    output_file: str = "analysis_results.json",
    confidence_threshold: float = 0.5
):
    """将分析结果保存为 JSON 文件
    
    Args:
        results: 分析结果字典
        output_file: 输出文件路径
        confidence_threshold: 置信度阈值
    """
    try:
        # 重新整理结果
        organized_results = reorganize_results(results, confidence_threshold)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(organized_results, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存结果失败: {str(e)}")


def print_summary(results: dict):
    """打印分析结果摘要
    
    Args:
        results: 分析结果字典
    """
    print("\n" + "=" * 80)
    print("📊 分析摘要")
    print("=" * 80)
    
    # 统计每种特征出现的次数
    feature_counts = {}
    images_with_valid_features = 0  # 有有效特征的图片（置信度 >= 阈值）
    images_with_low_confidence = 0  # 只有低置信度检测的图片
    images_without_features = 0  # 完全没有检测到任何特征的图片
    
    for image_name, data in results.items():
        if isinstance(data, dict) and "detections" in data:
            detections = data["detections"]
            valid_count = data.get("valid_count", 0)
            
            if valid_count > 0:
                # 有有效检测
                images_with_valid_features += 1
                # 统计特征（只统计有效的）
                for detection in detections:
                    if isinstance(detection, dict) and "feature" in detection:
                        # 这里需要根据置信度判断是否统计
                        # 但为了简化，我们统计所有检测到的特征
                        feature = detection["feature"]
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
            elif detections:
                # 有检测但都是低置信度
                images_with_low_confidence += 1
            else:
                # 完全没有检测到
                images_without_features += 1
    
    print(f"\n总图片数: {len(results)}")
    print(f"  - 检测到有效特征的图片: {images_with_valid_features}")
    print(f"  - 仅有低置信度检测的图片: {images_with_low_confidence}")
    print(f"  - 未检测到任何特征的图片: {images_without_features}")
    
    if feature_counts:
        print(f"\n特征检测统计（所有检测）:")
        for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {feature}: {count} 次")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 设置图片文件夹路径
    script_dir = Path(__file__).parent
    images_folder = script_dir / "test-images"
    
    # ==================== 配置参数 ====================
    USE_YOLO = True  # 是否使用 YOLO（True）或传统方法（False）
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值（0.0 - 1.0），可以调整这个值
    
    # Logo 检测配置
    ENABLE_LOGO_DETECTION = False  # 是否启用 Logo 检测（设置为 True 启用）
    LOGO_MODEL_PATH = None  # Logo 模型路径（如果有）
    # LOGO_MODEL_PATH = "./models/logodet3k_best.pt"  # 示例：指定模型路径
    # ===================================================
    
    print("\n" + "=" * 80)
    print("🖼️  批量图片分析工具")
    print("=" * 80)
    print(f"📂 图片文件夹: {images_folder}")
    print(f"🔧 检测方法: {'YOLO 深度学习' if USE_YOLO else '传统 OpenCV'}")
    print(f"📊 置信度阈值: {CONFIDENCE_THRESHOLD:.0%}\n")
    
    # 分析图片
    results = analyze_images_in_folder(
        str(images_folder), 
        use_yolo=USE_YOLO,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        enable_logo_detection=ENABLE_LOGO_DETECTION,
        logo_model_path=LOGO_MODEL_PATH
    )
    
    if results:
        # 打印摘要
        print_summary(results)
        
        # 保存结果到 JSON 文件
        output_file = script_dir / "analysis_results.json"
        save_results_to_json(results, str(output_file), CONFIDENCE_THRESHOLD)
    
    print("\n✅ 程序执行完成!\n")


if __name__ == "__main__":
    main()
