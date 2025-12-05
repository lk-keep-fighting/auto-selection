# Local Product Image Recognition

本项目实现了全本地化的产品图像识别系统，用于电商安全审核场景。支持两种使用模式：

1. **批量图片分析** - 分析本地文件夹中的图片
2. **🆕 浏览器自动化** - 自动打开浏览器、获取在线图片并识别

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

## 🚀 快速开始

### 方式一：浏览器自动化（推荐）

```bash
# 1. 安装依赖
pip install -e .

# 2. 安装 Playwright 浏览器 (macOS/Linux/Windows)
playwright install chromium

# 3. 检查环境
python3 check_environment.py

# 4. 运行浏览器自动化
python3 browser_automation.py
```

详细文档：
- 📖 [快速开始指南](QUICKSTART.md)
- 📖 [浏览器自动化完整文档](BROWSER_AUTOMATION_GUIDE.md)

### 方式二：批量分析本地图片

```bash
# 将图片放入 test-images/ 文件夹
python3 main.py
```

## 基本使用

### Python API

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

browser_automation.py      # 🆕 浏览器自动化主程序
config.json               # 浏览器自动化配置文件
check_environment.py      # 环境检查工具
main.py                   # 批量图片分析程序
```

The detectors are intentionally modular so additional categories can be layered
on without modifying the public entrypoint.

## 主要特性

✅ **全本地化** - 不依赖任何云服务或 API
✅ **浏览器自动化** - 基于 Playwright，更稳定更快速
✅ **登录状态保存** - 自动保存和恢复浏览器状态
✅ **多种检测器** - 支持 YOLO 和传统 OpenCV 方法
✅ **模块化设计** - 易于扩展新的检测类型
✅ **一键安装** - 无需关心浏览器驱动版本
✅ **配置灵活** - 通过 JSON 配置适配不同网站
