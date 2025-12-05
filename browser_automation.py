#!/usr/bin/env python3
"""浏览器自动化程序：自动登录、获取图片列表并进行识别 (基于 Playwright)"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import base64

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
import requests

from local_product_recognition import LocalProductImageRecognizer
from local_product_recognition.types import DetectionResult
from local_product_recognition.detectors.yolo import YOLODetector
from local_product_recognition.detectors.logo import BrandLogoDetector


class BrowserAutomation:
    """浏览器自动化类 (Playwright 实现)"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化浏览器自动化
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.recognizer = None
        self.yolo_detector = None
        self.logo_detector = None
        self.collected_brands = set()  # 收集的品牌名集合
        
        # 创建输出目录
        self.images_folder = Path(self.config["output"]["images_folder"])
        self.images_folder.mkdir(exist_ok=True, parents=True)
        
        # 初始化检测器
        self._init_detectors()
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ 配置文件不存在: {config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ 配置文件格式错误: {e}")
            raise
    
    def _init_detectors(self):
        """初始化图像检测器"""
        detection_config = self.config.get("detection", {})
        use_yolo = detection_config.get("use_yolo", True)
        confidence_threshold = detection_config.get("confidence_threshold", 0.5)
        enable_logo = detection_config.get("enable_logo_detection", False)
        logo_method = detection_config.get("logo_detection_method", "ocr")
        
        if use_yolo:
            print(f"🤖 初始化 YOLO 检测器 (置信度阈值: {confidence_threshold:.0%})")
            self.yolo_detector = YOLODetector(
                model_name="yolov8n.pt",
                confidence_threshold=0.25,
                device="cpu"
            )
            
            if enable_logo and logo_method == "ocr":
                print(f"🏷️  初始化 OCR Logo 检测器 (基于文字识别)")
                # 初始化时使用空列表，后续会根据采集的品牌更新
                self.logo_detector = BrandLogoDetector(
                    similarity_threshold=0.55,
                    brand_keywords=[]  # 初始为空，后续动态添加
                )
        else:
            print(f"📊 初始化传统检测器")
            self.recognizer = LocalProductImageRecognizer()
    
    def init_browser(self):
        """初始化浏览器"""
        browser_config = self.config.get("browser", {})
        
        try:
            print("🌐 正在启动浏览器...")
            
            # 启动 Playwright
            self.playwright = sync_playwright().start()
            
            # 浏览器启动参数
            launch_options = {
                "headless": browser_config.get("headless", False),
            }
            
            # 设置窗口大小
            window_size = browser_config.get("window_size", "1920,1080").split(",")
            viewport = {
                "width": int(window_size[0]),
                "height": int(window_size[1])
            }
            
            # 扩展（插件）目录
            extension_dir = Path("assets/kuajing-erp-plugin-v3").resolve()
            
            # 用户数据目录（用于保存登录状态 / 持久化上下文）
            user_data_dir = browser_config.get("user_data_dir")
            storage_state_file = None
            
            if extension_dir.exists():
                # 使用持久化上下文加载扩展（插件），扩展仅在非无头模式下工作
                launch_headless = browser_config.get("headless", False)
                if launch_headless:
                    print("⚠️ 扩展需要在非无头模式下运行，已强制关闭 headless")
                
                args = [
                    f"--disable-extensions-except={extension_dir}",
                    f"--load-extension={extension_dir}",
                ]
                
                # 准备 user_data_dir（必须存在）
                if not user_data_dir:
                    user_data_dir = "./browser_profile"
                user_data_path = Path(user_data_dir).resolve()
                user_data_path.mkdir(exist_ok=True, parents=True)
                
                self.context = self.playwright.chromium.launch_persistent_context(
                    user_data_dir=str(user_data_path),
                    headless=False,
                    args=args,
                )
                print(f"✅ 已加载插件: {extension_dir}")
            
            else:
                # 无扩展：正常启动浏览器 + 非持久化上下文
                self.browser = self.playwright.chromium.launch(**launch_options)
                
                if user_data_dir:
                    user_data_path = Path(user_data_dir).resolve()
                    user_data_path.mkdir(exist_ok=True, parents=True)
                    storage_state_file = user_data_path / "state.json"
                    
                    # 如果存在保存的状态，使用它
                    if storage_state_file.exists():
                        self.context = self.browser.new_context(
                            storage_state=str(storage_state_file),
                            viewport=viewport
                        )
                        print("✅ 已加载保存的浏览器状态")
                    else:
                        self.context = self.browser.new_context(viewport=viewport)
                else:
                    self.context = self.browser.new_context(viewport=viewport)
            
            # 创建页面
            self.page = self.context.new_page()
            # 设置视口大小（持久化上下文不支持在创建时设置）
            try:
                self.page.set_viewport_size(viewport)
            except Exception:
                pass
            
            print("✅ 浏览器启动成功")
        except Exception as e:
            print(f"❌ 浏览器启动失败: {e}")
            print("💡 提示: 请确保已安装 Playwright 浏览器")
            print("    安装命令: playwright install chromium")
            raise
    
    def open_url(self, url: str = None):
        """打开目标网址
        
        Args:
            url: 目标网址，如果为 None 则使用配置文件中的 targetUrl
        """
        if not self.page:
            raise RuntimeError("浏览器未初始化，请先调用 init_browser()")
        
        target_url = url or self.config.get("targetUrl")
        if not target_url:
            raise ValueError("未指定目标网址")
        
        print(f"🔗 正在打开网址: {target_url}")
        self.page.goto(target_url, wait_until="domcontentloaded")
        print("✅ 网页加载完成")
    
    def wait_for_user_confirmation(self, message: str = "请完成登录操作，然后在终端按 Enter 继续..."):
        """等待用户确认
        
        Args:
            message: 提示信息
        """
        print(f"\n⏸️  {message}")
        input()
        print("▶️  继续执行...")
    
    def save_login_state(self):
        """保存登录状态"""
        if not self.context:
            return
        
        browser_config = self.config.get("browser", {})
        user_data_dir = browser_config.get("user_data_dir")
        
        if user_data_dir:
            user_data_path = Path(user_data_dir).resolve()
            user_data_path.mkdir(exist_ok=True, parents=True)
            storage_state_file = user_data_path / "state.json"
            
            # 保存浏览器状态（包括 cookies, localStorage 等）
            self.context.storage_state(path=str(storage_state_file))
            print(f"💾 登录状态已保存到: {storage_state_file}")
        else:
            print("⚠️  未配置 user_data_dir，无法保存登录状态")
    
    def load_login_state(self) -> bool:
        """加载登录状态（Playwright 在创建 context 时自动加载）
        
        Returns:
            是否成功加载
        """
        # Playwright 在 init_browser 时已经加载了状态
        return True
    
    def get_image_list(self) -> List[Dict]:
        """获取页面中的图片列表
        
        Returns:
            图片信息列表，每个元素包含 url, asin, title 等信息
        """
        if not self.page:
            raise RuntimeError("浏览器未初始化")
        
        selectors = self.config.get("selectors", {})
        image_list_selector = selectors.get("imageList")
        card_selector = selectors.get("cardItem")
        image_selector = selectors.get("imageItem")
        asin_selector = selectors.get("asinSelector")
        title_selector = selectors.get("titleSelector")
        brand_selector = selectors.get("brandSelector")  # 新增品牌选择器
        
        print("\n🔍 正在查找图片列表...")
        
        try:
            # 等待图片列表容器加载（Playwright 自动等待）
            self.page.wait_for_selector(image_list_selector, timeout=10000)
            
            # 滚动页面以加载所有图片
            self._scroll_page()
            
            # 查找所有商品卡片
            cards = self.page.query_selector_all(card_selector)
            print(f"📦 找到 {len(cards)} 个商品")
            
            images_info = []
            
            for idx, card in enumerate(cards, 1):
                try:
                    # 提取图片 URL（从 style 属性中提取背景图片）
                    img_element = card.query_selector(image_selector)
                    if not img_element:
                        continue
                    
                    # 从 style 属性中提取背景图片 URL
                    style = img_element.get_attribute("style") or ""
                    img_url = None
                    
                    # 解析 background: url("...") 格式
                    import re
                    match = re.search(r'url\(["\']?(https?://[^"\')]+)["\']?\)', style)
                    if match:
                        img_url = match.group(1)
                    
                    # 如果没有找到，尝试 src 或 data-src 属性
                    if not img_url:
                        img_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")
                    
                    # 提取 ASIN
                    asin = ""
                    asin_element = card.query_selector(asin_selector)
                    if asin_element:
                        asin = asin_element.inner_text().strip()
                    
                    # 提取标题
                    title = ""
                    title_element = card.query_selector(title_selector)
                    if title_element:
                        title = title_element.get_attribute("title") or title_element.inner_text().strip()
                    
                    # 提取品牌名
                    brand = ""
                    if brand_selector:
                        # 尝试查找包含 "品牌:" 的段落
                        brand_paragraphs = card.query_selector_all("p.flex-center")
                        for p in brand_paragraphs:
                            text = p.inner_text()
                            if "品牌:" in text or "Brand:" in text.lower():
                                brand_element = p.query_selector(".over-ellipsis.text-black.sub-title")
                                if brand_element:
                                    brand = brand_element.inner_text().strip()
                                    break
                    
                    if img_url:
                        images_info.append({
                            "index": idx,
                            "url": img_url,
                            "asin": asin,
                            "title": title,
                            "brand": brand  # 新增品牌信息
                        })
                        print(f"  [{idx}] {asin or 'N/A'} - {brand or 'N/A'} - {title[:40] if title else 'N/A'}...")
                    
                except Exception as e:
                    print(f"  ⚠️  提取第 {idx} 个商品信息失败: {e}")
                    continue
            
            print(f"\n✅ 成功提取 {len(images_info)} 个图片信息")
            
            # 不再进行全局品牌收集，每张图片在分析阶段使用自身的 brand 进行匹配
            return images_info
            
        except PlaywrightTimeoutError:
            print(f"❌ 等待元素超时: {image_list_selector}")
            return []
        except Exception as e:
            print(f"❌ 获取图片列表失败: {e}")
            return []
    
    def _upgrade_image_resolution(self, img_url: str) -> str:
        """
        升级图片分辨率
        将 _US200_.jpg 替换为 _US600_.jpg 以获得更高分辨率
        
        Args:
            img_url: 原始图片URL
            
        Returns:
            升级后的图片URL
        """
        if not img_url:
            return img_url
        
        # 检测并替换不同的分辨率标识
        resolution_patterns = [
            ('_US200_', '_US600_'),
            ('_SX200_', '_SX600_'),
            ('_SY200_', '_SY600_'),
            ('_AC_US200_', '_AC_US600_'),
            ('.US200.', '.US600.'),
        ]
        
        upgraded_url = img_url
        for old_pattern, new_pattern in resolution_patterns:
            if old_pattern in upgraded_url:
                upgraded_url = upgraded_url.replace(old_pattern, new_pattern)
                print(f"      🔍 升级分辨率: {old_pattern} → {new_pattern}")
                break
        
        return upgraded_url
    
    def _scroll_page(self, scroll_pause: float = 1.0, max_scrolls: int = 5):
        print("📜 滚动页面加载内容...")
        
        last_height = self.page.evaluate("document.body.scrollHeight")
        scroll_count = 0
        
        while scroll_count < max_scrolls:
            # 滚动到页面底部
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(scroll_pause)
            
            # 计算新的页面高度
            new_height = self.page.evaluate("document.body.scrollHeight")
            
            if new_height == last_height:
                break
            
            last_height = new_height
            scroll_count += 1
        
        # 滚动回顶部
        self.page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
    
    def download_image(self, img_url: str, save_path: Path, max_retries: int = None, timeout: int = None) -> bool:
        """下载图片
        
        Args:
            img_url: 图片URL
            save_path: 保存路径
            max_retries: 最大重试次数（默认从配置读取）
            timeout: 超时时间/秒（默认从配置读取）
            
        Returns:
            是否成功下载
        """
        # 从配置读取默认值
        if max_retries is None:
            max_retries = self.config.get("output", {}).get("download_max_retries", 3)
        if timeout is None:
            timeout = self.config.get("output", {}).get("download_timeout", 30)
        for attempt in range(max_retries):
            try:
                # 如果是 base64 图片
                if img_url.startswith("data:image"):
                    header, encoded = img_url.split(",", 1)
                    data = base64.b64decode(encoded)
                    with open(save_path, 'wb') as f:
                        f.write(data)
                    return True
                
                # 普通 URL 图片
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Connection': 'keep-alive',
                }
                
                # 增加超时时间，添加重试
                response = requests.get(
                    img_url, 
                    headers=headers, 
                    timeout=timeout,  # 使用配置的超时时间
                    stream=True,  # 使用流式下载
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # 流式写入文件
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return True
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间: 2s, 4s, 6s
                    print(f"    ⏳ 下载超时，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ 下载失败: 连接超时（已重试{max_retries}次）")
                    return False
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"    ⏳ 连接错误，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ 下载失败: 网络连接错误")
                    return False
                    
            except requests.exceptions.HTTPError as e:
                # HTTP 错误通常不需要重试（如404）
                print(f"    ❌ 下载失败: HTTP {e.response.status_code}")
                return False
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"    ⏳ 下载出错，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {type(e).__name__}")
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ 下载失败: {type(e).__name__}: {str(e)[:100]}")
                    return False
        
        return False
    
    def analyze_images(self, images_info: List[Dict]) -> Dict:
        """分析图片列表
        
        Args:
            images_info: 图片信息列表
            
        Returns:
            分析结果字典
        """
        print("\n🔬 开始分析图片...")
        print("=" * 80)
        
        detection_config = self.config.get("detection", {})
        confidence_threshold = detection_config.get("confidence_threshold", 0.5)
        output_config = self.config.get("output", {})
        save_images = output_config.get("save_images", True)
        upgrade_resolution = output_config.get("upgrade_image_resolution", True)
        
        all_results = []
        
        for img_info in images_info:
            idx = img_info["index"]
            img_url = img_info["url"]
            asin = img_info.get("asin", "unknown")
            title = img_info.get("title", "")
            brand = img_info.get("brand", "")  # 页面采集的品牌
            
            # 升级图片分辨率
            if upgrade_resolution:
                upgraded_url = self._upgrade_image_resolution(img_url)
                if upgraded_url != img_url:
                    img_url = upgraded_url
                    img_info["url"] = upgraded_url  # 更新保存的URL
            
            print(f"\n[{idx}/{len(images_info)}] 分析: {asin}")
            if brand:
                print(f"  品牌: {brand}")
            print(f"  标题: {title[:60]}..." if title else "  标题: N/A")
            
            # 下载图片
            img_filename = f"{idx:03d}_{asin}.jpg"
            img_path = self.images_folder / img_filename
            
            if save_images:
                print(f"  📥 下载图片...")
                if not self.download_image(img_url, img_path):
                    all_results.append({
                        **img_info,
                        "error": "下载失败"
                    })
                    continue
            
            # 分析图片
            try:
                if self.yolo_detector:
                    # 使用 YOLO
                    import cv2
                    
                    if save_images:
                        img = cv2.imread(str(img_path))
                    else:
                        # 从 URL 直接加载
                        import numpy as np
                        response = requests.get(img_url, timeout=10)
                        img_array = np.frombuffer(response.content, np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    results: List[DetectionResult] = self.yolo_detector.detect_all(img)
                    
                    # 如果启用了 Logo 检测，追加 Logo 检测结果
                    if self.logo_detector:
                        try:
                            # 仅使用当前图片关联的品牌进行 OCR 匹配
                            if brand:
                                original_keywords = self.logo_detector.brand_keywords
                                page_brand_keywords = self.logo_detector._prepare_brand_keywords(
                                    [brand],
                                    []
                                )
                                # 只用当前商品的品牌关键字，不再混入全局品牌
                                self.logo_detector.brand_keywords = page_brand_keywords or []
                            else:
                                original_keywords = None
                            
                            logo_result = self.logo_detector.detect(img)
                            
                            # 恢复原始关键字列表（仅在之前保存过时）
                            if original_keywords is not None:
                                self.logo_detector.brand_keywords = original_keywords
                            
                            if logo_result:
                                results.append(logo_result)
                                # 调试信息：显示 Logo 检测详情
                                if logo_result.details:
                                    detected_brand = logo_result.details.get('brand')
                                    # 检查是否匹配错误的品牌
                                    if brand and detected_brand and detected_brand.upper() != brand.upper():
                                        print(f"    ⚠️  品牌匹配可能有误: 页面='{brand}', OCR='{detected_brand}'")
                                    print(f"  🔍 Logo 检测详情: method={logo_result.details.get('method')}, brand={detected_brand}, text={logo_result.details.get('recognized_text')}")
                            else:
                                print(f"  ℹ️  Logo 检测未发现结果")
                        except Exception as logo_err:
                            # Logo 检测失败，记录但不阻塞流程
                            print(f"  ⚠️  Logo 检测失败: {logo_err}")
                else:
                    # 使用传统方法
                    results: List[DetectionResult] = self.recognizer.analyze(str(img_path))
                
                # 格式化结果
                formatted_results = []
                valid_detections = 0
                ocr_results = []  # 存储 OCR 识别结果
                
                # 显示结果
                if results:
                    for detection in results:
                        # 基础检测信息
                        detection_data = {
                            "feature": detection.feature.value,
                            "confidence": round(detection.confidence, 4)
                        }
                        
                        # 如果有详细信息，也保存
                        if detection.details:
                            detection_data["details"] = detection.details
                            
                            # 提取 OCR 相关结果
                            # 1. 如果是 Logo 检测且使用了 OCR
                            if detection.feature.value == "brand_logo" and detection.details.get("method") == "ocr":
                                # 处理匹配的品牌
                                if detection.details.get("brand"):
                                    ocr_results.append({
                                        "type": "brand_logo",
                                        "brand": detection.details.get("brand", ""),
                                        "recognized_text": detection.details.get("recognized_text", ""),
                                        "confidence": round(detection.confidence, 4),
                                        "bounding_box": detection.details.get("bounding_box", []),
                                        "match_type": detection.details.get("match_type", "full")
                                    })
                                # 处理未匹配的 OCR 文本
                                elif detection.details.get("recognized_texts"):
                                    for text in detection.details.get("recognized_texts", []):
                                        ocr_results.append({
                                            "type": "unmatched_text",
                                            "recognized_text": text,
                                            "confidence": round(detection.confidence, 4),
                                            "match_type": "unmatched"
                                        })
                            
                            # 2. 如果包含 recognized_text 字段（通用 OCR 识别）
                            elif "recognized_text" in detection.details:
                                ocr_results.append({
                                    "type": detection.feature.value,
                                    "recognized_text": detection.details.get("recognized_text", ""),
                                    "confidence": round(detection.confidence, 4),
                                    "bounding_box": detection.details.get("bounding_box", []),
                                    "details": {k: v for k, v in detection.details.items() if k not in ["recognized_text", "bounding_box"]}
                                })
                        
                        formatted_results.append(detection_data)
                        if detection.confidence >= confidence_threshold:
                            valid_detections += 1
                    
                    # 调试信息：显示 OCR 结果收集情况
                    print(f"  📊 OCR 结果收集: 总检测={len(results)}, OCR结果={len(ocr_results)}")
                    
                    # 显示检测结果
                    if valid_detections > 0:
                        print(f"  ✅ 检测到 {valid_detections} 个有效特征:")
                        for detection in results:
                            if detection.confidence >= confidence_threshold:
                                feature_desc = detection.feature.value
                                # 如果是 Logo 检测，显示检测到的品牌
                                if detection.feature.value == "brand_logo" and detection.details:
                                    detected_brand = detection.details.get("brand", "")
                                    method = detection.details.get("method", "")
                                    if detected_brand:
                                        feature_desc = f"{feature_desc} ({detected_brand} via {method})"
                                        # 对比页面品牌与检测品牌
                                        if brand and detected_brand.upper() == brand.upper():
                                            feature_desc += " ✅匹配"
                                        elif brand:
                                            feature_desc += f" ⚠️与页面品牌不匹配({brand})"
                                print(f"     - {feature_desc}: {detection.confidence:.2%}")
                        
                        # 显示 OCR 识别结果
                        if ocr_results:
                            print(f"  📝 OCR 识别结果 ({len(ocr_results)} 个):")
                            for ocr in ocr_results:
                                # 处理不同类型的 OCR 结果
                                if ocr.get('type') == 'brand_logo' and ocr.get('brand'):
                                    print(f"     - 品牌: {ocr['brand']}, 文本: '{ocr.get('recognized_text', '')}', 置信度: {ocr['confidence']:.2%}")
                                elif ocr.get('type') == 'unmatched_text':
                                    print(f"     - 未匹配文本: '{ocr.get('recognized_text', '')}', 置信度: {ocr['confidence']:.2%}")
                                else:
                                    # 其他类型
                                    print(f"     - 文本: '{ocr.get('recognized_text', '')}', 置信度: {ocr['confidence']:.2%}")
                    else:
                        print(f"  ℹ️  所有检测都低于阈值 {confidence_threshold:.0%}")
                else:
                    print(f"  ℹ️  未检测到任何特征")
                
                all_results.append({
                    **img_info,
                    "image_file": img_filename if save_images else None,
                    "detections": formatted_results,
                    "ocr_results": ocr_results if ocr_results else None,  # 新增 OCR 结果
                    "valid_count": valid_detections
                })
                
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                all_results.append({
                    **img_info,
                    "error": str(e)
                })
        
        print("\n" + "=" * 80)
        print(f"✨ 分析完成! 共处理 {len(images_info)} 张图片\n")
        
        return self._organize_results(all_results, confidence_threshold)
    
    def _organize_results(self, results: List[Dict], confidence_threshold: float) -> Dict:
        """整理分析结果
        
        Args:
            results: 原始结果列表
            confidence_threshold: 置信度阈值
            
        Returns:
            整理后的结果字典
        """
        passed = []
        detected = []
        
        for item in results:
            if "error" in item:
                detected.append(item)
                continue
            
            valid_count = item.get("valid_count", 0)
            
            if valid_count > 0:
                # 有有效检测
                valid_features = []
                low_confidence = []
                
                for detection in item.get("detections", []):
                    if detection["confidence"] >= confidence_threshold:
                        valid_features.append(detection)
                    else:
                        low_confidence.append(detection)
                
                detected.append({
                    "asin": item.get("asin"),
                    "title": item.get("title"),
                    "brand": item.get("brand"),  # 新增品牌
                    "image_url": item.get("url"),
                    "image_file": item.get("image_file"),
                    "features": valid_features,
                    "ocr_results": item.get("ocr_results"),  # 新增 OCR 结果
                    "low_confidence_detections": low_confidence if low_confidence else None
                })
            else:
                # 没有有效检测
                passed_item = {
                    "asin": item.get("asin"),
                    "title": item.get("title"),
                    "brand": item.get("brand"),  # 新增品牌
                    "image_url": item.get("url"),
                    "image_file": item.get("image_file"),
                    "ocr_results": item.get("ocr_results")  # 新增 OCR 结果
                }
                
                if item.get("detections"):
                    passed_item["low_confidence_detections"] = item["detections"]
                
                passed.append(passed_item)
        
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
    
    def save_results(self, results: Dict):
        """保存分析结果
        
        Args:
            results: 分析结果字典
        """
        output_file = self.config.get("output", {}).get("results_file", "browser_analysis_results.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def collect_passed_products(self, passed_items: List[Dict]):
        """对通过检测的商品进行自动采集
        
        Args:
            passed_items: 通过检测的商品列表
        """
        if not passed_items:
            print("\n✅ 没有需要采集的商品")
            return
        
        print(f"\n📦 开始采集 {len(passed_items)} 个通过检测的商品...")
        
        collected_count = 0
        failed_count = 0
        
        for idx, item in enumerate(passed_items, 1):
            asin = item.get("asin", "")
            title = item.get("title", "")
            brand = item.get("brand", "")
            
            print(f"\n[{idx}/{len(passed_items)}] 采集商品: {asin}")
            if brand:
                print(f"  品牌: {brand}")
            print(f"  标题: {title[:60]}...")
            
            try:
                # 1. 在当前页面找到对应的 ASIN 元素
                asin_selector = f'.asin .text-black:text("{asin}")'
                asin_element = self.page.query_selector(asin_selector)
                
                if not asin_element:
                    print(f"  ❌ 未找到 ASIN 元素")
                    failed_count += 1
                    continue
                
                # 2. 找到同级的亚马逊链接
                parent = asin_element.evaluate('el => el.closest(".asin")')
                link_element = self.page.query_selector(f'.asin:has(.text-black:text("{asin}")) a[href*="/dp/"]')
                
                if not link_element:
                    print(f"  ❌ 未找到产品链接")
                    failed_count += 1
                    continue
                
                # 3. 获取链接 URL
                product_url = link_element.get_attribute("href")
                print(f"  🔗 打开商品页面...")
                
                # 4. 在新标签页打开
                new_page = self.context.new_page()
                new_page.goto(product_url, wait_until="domcontentloaded")
                
                # 5. 等待页面加载完成（等待插件注入）
                print(f"  ⏳ 等待插件注入...")
                
                # 多次重试查找按钮，因为插件可能需要时间加载
                max_retries = 3
                retry_interval = 3
                button_clicked = False
                
                for retry in range(max_retries):
                    if retry > 0:
                        print(f"  🔄 重试 {retry}/{max_retries-1}...")
                    
                    time.sleep(retry_interval)
                    
                    # 保存截图用于调试
                    if retry == max_retries - 1:  # 最后一次重试才保存截图
                        screenshot_path = f"debug_screenshot_{asin}.png"
                        new_page.screenshot(path=screenshot_path)
                        print(f"  📸 已保存截图: {screenshot_path}")
                
                    # 6. 查找"采集此产品"按钮（直接通过class查找）
                    try:
                        result = new_page.evaluate("""
                            () => {
                                const log = [];
                                let buttonFound = false;
                                let foundButton = null;
                                
                                function searchInContext(root, contextName, depth) {
                                    if (depth === undefined) depth = 0;
                                    if (depth > 15) return null;
                                    
                                    // Strategy A: Direct class search
                                    const directButton = root.querySelector('.earth-wxt-collect-button');
                                    if (directButton) {
                                        log.push(contextName + ': Found by class .earth-wxt-collect-button');
                                        return directButton;
                                    }
                                    
                                    // Strategy B: Search by text (for elements with text content)
                                    const allElements = root.querySelectorAll('*');
                                    log.push(contextName + ': Searching ' + allElements.length + ' elements');
                                    
                                    for (let i = 0; i < allElements.length; i++) {
                                        const el = allElements[i];
                                        const text = (el.textContent || '').trim();
                                        
                                        // Check class name contains 'collect' or 'earth-wxt'
                                        const className = el.className || '';
                                        if (className.indexOf('collect') > -1 || className.indexOf('earth-wxt') > -1) {
                                            log.push('  Found by class: ' + el.tagName + ' class=' + className);
                                            return el;
                                        }
                                        
                                        // Check text contains collect keywords
                                        if (text.indexOf('\u91c7\u96c6\u6b64\u5546\u54c1') > -1 || 
                                            text.indexOf('\u91c7\u96c6\u6b64\u4ea7\u54c1') > -1 || 
                                            text.indexOf('\u91c7\u96c6') > -1) {
                                            
                                            log.push('  Found text match: ' + el.tagName + ', text: ' + text.substring(0, 50));
                                            
                                            const isClickable = el.tagName === 'BUTTON' || 
                                                              el.tagName === 'A' ||
                                                              el.tagName === 'DIV' ||
                                                              el.tagName === 'SPAN' ||
                                                              el.onclick !== null ||
                                                              el.classList.contains('button') ||
                                                              el.classList.contains('btn') ||
                                                              el.getAttribute('role') === 'button' ||
                                                              el.style.cursor === 'pointer';
                                            
                                            if (isClickable) {
                                                log.push('    -> Clickable!');
                                                return el;
                                            } else {
                                                const clickableChild = el.querySelector('button, a, [role=button], [onclick]');
                                                if (clickableChild) {
                                                    log.push('    -> Found clickable child: ' + clickableChild.tagName);
                                                    return clickableChild;
                                                }
                                            }
                                        }
                                        
                                        // Recursive search in Shadow DOM
                                        if (el.shadowRoot) {
                                            const found = searchInContext(el.shadowRoot, contextName + ' > ShadowRoot', depth + 1);
                                            if (found) return found;
                                        }
                                    }
                                    return null;
                                }
                                
                                log.push('=== Strategy 1: Search in main document ===');
                                foundButton = searchInContext(document, 'MainDoc');
                                
                                if (foundButton) {
                                    log.push('SUCCESS: Button found in main document');
                                    buttonFound = true;
                                } else {
                                    log.push('=== Strategy 2: Check if plugin loaded ===');
                                    // Check for any element with earth-wxt or collect in class
                                    const pluginElements = document.querySelectorAll('[class*=earth], [class*=wxt], [class*=collect]');
                                    log.push('Found ' + pluginElements.length + ' potential plugin elements');
                                    for (let i = 0; i < Math.min(10, pluginElements.length); i++) {
                                        const el = pluginElements[i];
                                        log.push('  [' + (i+1) + '] ' + el.tagName + ' class="' + el.className + '"');
                                    }
                                }
                                
                                if (foundButton) {
                                    foundButton.style.display = 'block';
                                    foundButton.style.visibility = 'visible';
                                    foundButton.style.opacity = '1';
                                    foundButton.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                    foundButton.click();
                                } else {
                                    log.push('=== FAILED: Plugin may not be loaded ===');
                                    log.push('Waiting longer and checking again...');
                                }
                                
                                return { success: buttonFound, log: log };
                            }
                        """)
                        
                        # 打印调试日志
                        if retry == 0 or result.get('success') or retry == max_retries - 1:
                            print(f"  🔍 搜索结果:")
                            for log_line in result.get('log', []):
                                print(f"     {log_line}")
                        
                        if result.get('success'):
                            print(f"  ✅ 已点击采集按钮")
                            
                            # 等待成功提示出现
                            try:
                                success_message = new_page.wait_for_selector(
                                    '.earth-wxt-message--success',
                                    timeout=5000,
                                    state='visible'
                                )
                                
                                if success_message:
                                    # 获取提示文字
                                    message_text = new_page.evaluate('''
                                        () => {
                                            const msg = document.querySelector('.earth-wxt-message--success .earth-wxt-message__content');
                                            return msg ? msg.textContent : '';
                                        }
                                    ''')
                                    print(f"  ✅ 采集成功: {message_text}")
                                else:
                                    print(f"  ⚠️  按钮已点击，但未看到成功提示")
                            except Exception as msg_err:
                                print(f"  ⚠️  按钮已点击，等待提示超时: {msg_err}")
                            
                            time.sleep(2)  # 等待提示消失
                            collected_count += 1
                            button_clicked = True
                            break  # 成功后退出重试循环
                        elif retry == max_retries - 1:
                            print(f"  ❌ 未找到可点击的采集按钮")
                            print(f"  💡 提示: 请查看截图 {screenshot_path} 确认按钮位置")
                            failed_count += 1
                            
                    except Exception as btn_err:
                        if retry == max_retries - 1:
                            print(f"  ❌ 查找按钮异常: {btn_err}")
                            import traceback
                            traceback.print_exc()
                            failed_count += 1
                            break

                # 7. 关闭标签页
                new_page.close()
                
            except Exception as e:
                print(f"  ❌ 采集失败: {e}")
                failed_count += 1
                # 关闭可能打开的页面
                try:
                    if 'new_page' in locals() and not new_page.is_closed():
                        new_page.close()
                except:
                    pass
        
        print(f"\n🎉 采集任务完成")
        print(f"  - ✅ 成功: {collected_count}")
        print(f"  - ❌ 失败: {failed_count}")
    
    def close(self):
        """关闭浏览器"""
        if self.context:
            print("\n🔚 关闭浏览器...")
            self.context.close()
            self.context = None
        
        if self.browser:
            self.browser.close()
            self.browser = None
        
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        
        print("✅ 浏览器已关闭")
    
    def run(self):
        """运行完整流程"""
        try:
            # 1. 初始化浏览器
            self.init_browser()
            
            # 2. 打开网址
            self.open_url()
            
            # 3. 等待用户登录
            self.wait_for_user_confirmation("请完成登录操作（如果需要），然后按 Enter 继续...")
            
            # 4. 保存登录状态
            self.save_login_state()
            
            # 5. 获取图片列表
            images_info = self.get_image_list()
            
            if not images_info:
                print("⚠️  未找到任何图片，程序退出")
                return
            
            # 6. 分析图片
            results = self.analyze_images(images_info)
            
            # 7. 保存结果
            self.save_results(results)
            
            # 8. 打印摘要
            self._print_summary(results)
            
            # 9. 采集通过检测的商品
            passed_items = results.get("passed", [])
            if passed_items:
                print("\n" + "=" * 80)
                print("📦 开始自动采集通过检测的商品")
                print("=" * 80)
                self.collect_passed_products(passed_items)
            
        finally:
            # 关闭浏览器
            self.close()
    
    def _print_summary(self, results: Dict):
        """打印结果摘要"""
        summary = results.get("summary", {})
        
        print("\n" + "=" * 80)
        print("📊 分析摘要")
        print("=" * 80)
        print(f"\n总商品数: {summary.get('total', 0)}")
        print(f"  - ✅ 通过（无敏感特征）: {summary.get('passed_count', 0)}")
        print(f"  - ⚠️  检测到敏感特征: {summary.get('detected_count', 0)}")
        print(f"\n置信度阈值: {summary.get('confidence_threshold', 0):.0%}")
        print("=" * 80)


def main():
    """主函数"""
    import sys
    
    # 支持命令行参数指定配置文件
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    
    print("\n" + "=" * 80)
    print("🤖 浏览器自动化图片识别系统 (Playwright)")
    print("=" * 80)
    print(f"📄 配置文件: {config_file}\n")
    
    automation = BrowserAutomation(config_file)
    automation.run()
    
    print("\n✅ 程序执行完成!\n")


if __name__ == "__main__":
    main()
