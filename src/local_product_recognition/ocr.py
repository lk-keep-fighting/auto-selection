"""Lightweight OCR helpers for brand logo text matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .image_utils import ensure_gray

# 尝试导入 pytesseract（如果可用）
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@dataclass(frozen=True)
class RecognizedText:
    """Represents a single block of text recognized in an image."""

    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h


class LightweightTextRecognizer:
    """A tiny OCR helper tailored for simple, high-contrast product logos.

    The recognizer is purposely lightweight so it can run locally without
    external dependencies such as Tesseract. It works best for uppercase words
    rendered with relatively clean fonts (e.g. Hershey fonts used in OpenCV).
    """

    def __init__(
        self,
        char_set: Sequence[str] | str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        template_size: int = 48,  # 增大模板尺寸，提高精度
        min_word_score: float = 0.35,  # 降低阈值，提高召回率
        min_char_score: float = 0.25,  # 降低单字符阈值
        use_tesseract: bool = True,  # 是否优先使用 Tesseract
    ) -> None:
        if isinstance(char_set, str):
            iterable = list(dict.fromkeys(char_set))
        else:
            iterable = list(dict.fromkeys(char_set))
        self.char_set = [c.upper() for c in iterable]
        self.template_size = template_size
        self.min_word_score = min_word_score
        self.min_char_score = min_char_score
        self.use_tesseract = use_tesseract and TESSERACT_AVAILABLE
        self._templates = self._build_templates()
        
        # 检查 Tesseract 可用性
        if self.use_tesseract:
            try:
                # 测试 Tesseract 是否工作
                test_img = np.zeros((100, 100), dtype=np.uint8)
                pytesseract.get_tesseract_version()
                print("\u2705 Tesseract OCR 可用，将使用高精度识别")
            except Exception as e:
                print(f"\u26a0\ufe0f  Tesseract 不可用: {e}\uff0c回退到轻量级模式")
                self.use_tesseract = False

    def detect(self, image: np.ndarray) -> List[RecognizedText]:
        """Return text regions detected in the provided image."""

        if not self._templates:
            return []
        
        # 如果 Tesseract 可用，优先使用
        if self.use_tesseract:
            tesseract_results = self._detect_with_tesseract(image)
            if tesseract_results:
                return tesseract_results
            # 如果 Tesseract 失败，回退到轻量级模式

        binary = self._prepare_binary(image)
        if binary is None:
            return []

        detections: List[RecognizedText] = []
        for x, y, w, h in self._find_text_regions(binary):
            region = binary[y : y + h, x : x + w]
            recognized = self._recognize_region(region)
            if recognized is None:
                continue
            text, score = recognized
            if score < self.min_word_score:
                continue
            detections.append(
                RecognizedText(
                    text=text,
                    confidence=float(score),
                    bounding_box=(int(x), int(y), int(w), int(h)),
                )
            )

        detections.sort(key=lambda det: (-det.confidence, det.bounding_box[1], det.bounding_box[0]))
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    def _detect_with_tesseract(self, image: np.ndarray) -> List[RecognizedText]:
        """使用 Tesseract OCR 进行高精度文字识别"""
        if not TESSERACT_AVAILABLE:
            return []
        
        try:
            # 图像预处理
            gray = ensure_gray(image)
            if gray is None:
                return []
            
            # 增强预处理：多种方法结合
            # 1. CLAHE 增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 2. 双边滤波去噪
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 3. 尝试多种二值化方法
            # Otsu 阈值
            _, binary1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 自适应阈值
            binary2 = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 使用多个 PSM 模式尝试识别
            psm_modes = [
                6,   # 单个文本块
                7,   # 单行文本
                11,  # 稀疏文本
                3,   # 自动页面分割
            ]
            
            all_detections = []
            
            # 对每种二值化结果和 PSM 模式尝试识别
            for binary in [binary1, binary2]:
                for psm in psm_modes:
                    # 配置 Tesseract
                    custom_config = f'--oem 3 --psm {psm}'
                    
                    try:
                        # 获取详细数据
                        data = pytesseract.image_to_data(
                            binary, 
                            output_type=pytesseract.Output.DICT, 
                            config=custom_config
                        )
                        
                        n_boxes = len(data['text'])
                        for i in range(n_boxes):
                            text = data['text'][i].strip()
                            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                            
                            # 过滤空文本和低置信度
                            if not text or conf < 20:
                                continue
                            
                            # 过滤单字符（除非高置信度）
                            if len(text) == 1 and conf < 50:
                                continue
                            
                            # 过滤纯数字（除非高置信度）
                            if text.isdigit() and len(text) < 3 and conf < 40:
                                continue
                            
                            # 过滤特殊字符
                            import re
                            # 保留字母数字和常见符号
                            cleaned_text = re.sub(r'[^A-Za-z0-9\s\-&]', '', text)
                            if not cleaned_text.strip():
                                continue
                            
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            
                            # 转换置信度到 0-1 范围
                            confidence = conf / 100.0
                            
                            all_detections.append({
                                'text': cleaned_text.strip().upper(),
                                'confidence': confidence,
                                'bbox': (int(x), int(y), int(w), int(h)),
                                'psm': psm
                            })
                    except Exception:
                        continue
            
            # 去重：合并相似的检测结果
            unique_detections = []
            seen_texts = set()
            
            # 按置信度排序
            all_detections.sort(key=lambda x: -x['confidence'])
            
            for det in all_detections:
                text = det['text']
                # 去除空格后比较
                text_no_space = text.replace(' ', '')
                
                # 检查是否与已有结果重复
                is_duplicate = False
                for seen in seen_texts:
                    seen_no_space = seen.replace(' ', '')
                    # 如果文本相同或包含关系
                    if (text_no_space == seen_no_space or 
                        text_no_space in seen_no_space or 
                        seen_no_space in text_no_space):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_texts.add(text)
                    x, y, w, h = det['bbox']
                    unique_detections.append(
                        RecognizedText(
                            text=text,
                            confidence=float(det['confidence']),
                            bounding_box=(x, y, w, h),
                        )
                    )
            
            # 按置信度排序
            unique_detections.sort(key=lambda det: (-det.confidence, det.bounding_box[1], det.bounding_box[0]))
            return unique_detections[:20]  # 最多返回 20 个结果
            
        except Exception as e:
            # Tesseract 失败，静默返回空列表，会回退到轻量级模式
            return []

    def _prepare_binary(self, image: np.ndarray) -> np.ndarray | None:
        gray = ensure_gray(image)
        if gray is None:
            return None
        
        # 增强图像预处理
        # 1. 自适应直方图均衡化，增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. 双边滤波，保持边缘同时去噪
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. 尝试多种二值化方法，选择最佳结果
        # 方法1: Otsu 阈值
        _, thresh1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法2: 自适应阈倿
        thresh2 = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 选择白色像素比例更合理的结果
        white_ratio1 = float(cv2.countNonZero(thresh1)) / float(thresh1.size)
        white_ratio2 = float(cv2.countNonZero(thresh2)) / float(thresh2.size)
        
        # 选择白色比例在 0.2-0.8 之间的结果
        if 0.2 <= white_ratio1 <= 0.8:
            thresh = thresh1
        elif 0.2 <= white_ratio2 <= 0.8:
            thresh = thresh2
        else:
            # 都不好，选择更接近 0.5 的
            thresh = thresh1 if abs(white_ratio1 - 0.5) < abs(white_ratio2 - 0.5) else thresh2
        
        # 如果白色像素超过 50%，反转（黑底白字 -> 白底黑字）
        white_ratio = float(cv2.countNonZero(thresh)) / float(thresh.size)
        if white_ratio > 0.5:
            thresh = cv2.bitwise_not(thresh)
        
        # 4. 形态学处理，去除小噪点
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise)
        
        return thresh

    def _find_text_regions(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = binary.shape[:2]
        
        # 增大水平连接核，更好地连接字符
        kernel_width = max(5, width // 30)  # 从 40 改为 30，更容易连接
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # 增加迭代次数
        
        # 再次闭运算，进一步连接
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel_vertical)
        
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 降低最小尺寸限制，提高召回率
            if w < 8 or h < 8:  # 从 12 降低到 8
                continue
            if w * h < 100:  # 从 200 降低到 100
                continue
            
            aspect = w / float(h)
            # 放宽纵横比限制
            if aspect < 0.3 and w < 20:  # 从 0.5 和 30 放宽
                continue
            
            # 过滤过大的区域（可能是背景）
            if w > width * 0.95 or h > height * 0.8:
                continue
            
            regions.append((x, y, w, h))

        regions.sort(key=lambda box: (box[1], box[0]))
        return regions[:50]  # 从 32 增加到 50，处理更多区域

    def _recognize_region(self, region: np.ndarray) -> Tuple[str, float] | None:
        cropped = self._crop_to_content(region)
        if cropped is None:
            return None

        mask = (cropped > 0).astype(np.uint8)
        if mask.size == 0:
            return None

        segments = self._segment_characters(mask)
        if not segments:
            return None

        chars: List[str] = []
        scores: List[float] = []
        for segment in segments:
            prepared = self._prepare_character(segment)
            if prepared is None:
                continue
            char, score = self._match_character(prepared)
            if char is None or score < self.min_char_score:
                continue
            chars.append(char)
            scores.append(score)

        if not chars:
            return None

        text = "".join(chars)
        avg_score = float(sum(scores) / len(scores))
        return text, avg_score

    def _segment_characters(self, mask: np.ndarray) -> List[np.ndarray]:
        """改进的字符分割，使用投影法和连通组件法结合"""
        # 方法1: 垂直投影法
        projection = mask.sum(axis=0)
        threshold = max(1, int(mask.shape[0] * 0.15))  # 从 0.2 降低到 0.15

        segments: List[np.ndarray] = []
        start = None
        for idx, value in enumerate(projection):
            if value > threshold:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    end = idx
                    if end - start >= 2:
                        left = max(0, start - 1)
                        right = min(mask.shape[1], end + 1)
                        segments.append(mask[:, left:right])
                    start = None
        if start is not None:
            end = mask.shape[1]
            if end - start >= 2:
                left = max(0, start - 1)
                segments.append(mask[:, left:end])
        
        # 如果投影法分割失败或结果太少，尝试连通组件法
        if len(segments) < 2:
            # 使用连通组件分析
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            component_segments = []
            
            for label in range(1, num_labels):  # 跳过背景（标签0）
                component_mask = (labels == label).astype(np.uint8)
                coords = cv2.findNonZero(component_mask)
                if coords is None:
                    continue
                x, y, w, h = cv2.boundingRect(coords)
                if w >= 2 and h >= 2:  # 过滤太小的组件
                    component_segments.append(mask[y:y+h, x:x+w])
            
            # 如果连通组件法找到更多分割，使用它
            if len(component_segments) > len(segments):
                segments = component_segments
                # 按 x 坐标排序
                segments.sort(key=lambda s: cv2.boundingRect(cv2.findNonZero(s))[0] if cv2.findNonZero(s) is not None else 0)
        
        return segments

    def _prepare_character(self, segment: np.ndarray) -> np.ndarray | None:
        # 验证 segment 是否有效
        if segment is None or segment.size == 0:
            return None
        if segment.shape[0] == 0 or segment.shape[1] == 0:
            return None
        
        try:
            cropped = self._crop_to_content((segment > 0).astype(np.uint8) * 255)
            if cropped is None:
                return None
            return self._resize_with_padding(cropped, self.template_size)
        except (ValueError, IndexError, cv2.error):
            # 如果处理失败，返回 None
            return None

    def _match_character(self, char_img: np.ndarray) -> Tuple[str | None, float]:
        char_vector = (char_img / 255.0).astype(np.float32)
        if not np.any(char_vector):
            return None, 0.0

        best_char: str | None = None
        best_score = 0.0
        for char, variants in self._templates.items():
            for template in variants:
                score = self._cosine_similarity(char_vector, template)
                if score > best_score:
                    best_score = score
                    best_char = char
        return best_char, best_score

    def _build_templates(self) -> dict[str, List[np.ndarray]]:
        # 增加更多字体和变体，提高识别鲁棒性
        fonts = [
            # 原有字体
            (cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2),
            (cv2.FONT_HERSHEY_DUPLEX, 1.0, 2),
            (cv2.FONT_HERSHEY_PLAIN, 1.5, 2),
            # 新增字体变体
            (cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2),
            (cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2),
            (cv2.FONT_HERSHEY_DUPLEX, 1.2, 2),
            (cv2.FONT_HERSHEY_COMPLEX, 1.0, 2),
            (cv2.FONT_HERSHEY_TRIPLEX, 1.0, 2),
            # 粗体变体
            (cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3),
            (cv2.FONT_HERSHEY_DUPLEX, 1.0, 3),
        ]
        templates: dict[str, List[np.ndarray]] = {}
        canvas_size = self.template_size * 2

        for raw_char in self.char_set:
            char = raw_char.upper()
            variants: List[np.ndarray] = []
            for font_face, scale, thickness in fonts:
                canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                baseline = canvas_size // 2 + int(scale * 4)
                cv2.putText(
                    canvas,
                    char,
                    (4, baseline),
                    font_face,
                    scale,
                    255,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cropped = self._crop_to_content(canvas)
                if cropped is None:
                    continue
                prepared = self._resize_with_padding(cropped, self.template_size)
                variants.append((prepared / 255.0).astype(np.float32))
            if variants:
                templates[char] = variants
        return templates

    @staticmethod
    def _resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        scale = min((target_size - 4) / h, (target_size - 4) / w)
        scale = max(scale, 0.1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        
        try:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            # 如果 resize 失败，返回空画布
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 确保不超出边界
        y_end = min(y_offset + new_h, target_size)
        x_end = min(x_offset + new_w, target_size)
        actual_h = y_end - y_offset
        actual_w = x_end - x_offset
        
        if actual_h > 0 and actual_w > 0:
            try:
                canvas[y_offset:y_end, x_offset:x_end] = resized[:actual_h, :actual_w]
            except (ValueError, IndexError):
                # 如果仍然失败，返回空画布
                pass
        
        return canvas

    @staticmethod
    def _crop_to_content(image: np.ndarray) -> np.ndarray | None:
        if np.count_nonzero(image) == 0:
            return None
        coords = cv2.findNonZero(image)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_vec = a.flatten()
        b_vec = b.flatten()
        denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / denom)
