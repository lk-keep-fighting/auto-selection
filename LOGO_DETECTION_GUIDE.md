# Logo 检测模型配置指南

本项目已集成 Logo 检测功能，支持使用 YOLOv8 模型检测品牌 logo。

## 🚀 快速开始

### 选项 1: 使用 Roboflow (推荐 - 最简单)

1. **注册 Roboflow 账号**
   - 访问: https://roboflow.com/
   - 注册免费账号

2. **获取 API Key**
   - 登录后访问: https://app.roboflow.com/settings/api
   - 复制你的 API Key

3. **安装 Roboflow 库**
   ```bash
   pip install roboflow
   ```

4. **在 main.py 中配置**
   ```python
   # 在 main() 函数中修改:
   ENABLE_LOGO_DETECTION = True
   LOGO_MODEL_PATH = None  # 使用 Roboflow 时不需要
   
   # 然后修改 analyze_images_in_folder 调用，添加 Roboflow 参数
   # 或直接在 logo_yolo.py 的 YOLOLogoDetector 初始化时设置:
   # logo_detector = YOLOLogoDetector(
   #     use_roboflow=True,
   #     roboflow_api_key="你的API_KEY"
   # )
   ```

### 选项 2: 手动下载预训练模型

1. **从 Hugging Face 下载** (如果可用)
   ```bash
   # 创建模型目录
   mkdir -p models
   
   # 使用 wget 或浏览器下载模型文件
   # 将下载的 .pt 文件放到 models/ 目录
   ```

2. **在 main.py 中配置**
   ```python
   ENABLE_LOGO_DETECTION = True
   LOGO_MODEL_PATH = "./models/logodet3k_best.pt"  # 你的模型文件路径
   ```

### 选项 3: 训练自定义模型

如果你只需要检测特定品牌的 logo:

1. **准备数据集**
   ```bash
   # 数据集结构:
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/
   ```

2. **创建 data.yaml**
   ```yaml
   train: ./dataset/train/images
   val: ./dataset/val/images
   nc: 10  # 品牌数量
   names: ['HP', 'Apple', 'Nike', 'Adidas', ...]  # 品牌名称列表
   ```

3. **训练模型**
   ```bash
   yolo train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```

4. **使用训练好的模型**
   ```python
   ENABLE_LOGO_DETECTION = True
   LOGO_MODEL_PATH = "./runs/detect/train/weights/best.pt"
   ```

## 📦 支持的 Logo 检测模型

### LogoDet-3K
- **品牌数量**: 3,000+
- **数据集**: 200,000+ 标注对象
- **来源**: https://github.com/Wangjing1551/LogoDet-3K-Dataset
- **适合**: 需要检测大量品牌

### 自定义模型
- **品牌数量**: 根据你的需求
- **适合**: 只需检测特定品牌（推荐）

## 🔧 配置说明

在 `main.py` 中找到配置区域:

```python
# ==================== 配置参数 ====================
USE_YOLO = True                    # 使用 YOLO
CONFIDENCE_THRESHOLD = 0.5         # 置信度阈值
ENABLE_LOGO_DETECTION = False      # 👈 改为 True 启用 Logo 检测
LOGO_MODEL_PATH = None             # 👈 模型路径（如果有）
# ==================================================
```

## 📊 检测结果

启用 Logo 检测后，JSON 输出会包含 Logo 信息:

```json
{
  "image": "product.jpg",
  "features": [
    {
      "feature": "brand_logo",
      "confidence": 0.85,
      "brand": "HP",  // 👈 检测到的品牌名称
      "bounding_box": [100, 50, 200, 150]
    }
  ]
}
```

## ❓ 常见问题

### Q: Logo 检测不准确怎么办?
A: 
1. 提高 `confidence_threshold` (如 0.5 → 0.7)
2. 使用更大的模型 (yolov8m.pt, yolov8l.pt)
3. 训练自定义模型，只包含你需要的品牌

### Q: 可以同时检测多个 logo 吗?
A: 可以！`detect()` 方法会返回图片中所有检测到的 logo。

### Q: 支持哪些品牌?
A: 取决于你使用的模型。LogoDet-3K 支持 3000+ 品牌，自定义模型支持你训练的品牌。

## 📝 示例代码

```python
from local_product_recognition.detectors.logo_yolo import YOLOLogoDetector
import cv2

# 初始化检测器
detector = YOLOLogoDetector(
    model_path="./models/logodet3k_best.pt",
    confidence_threshold=0.5
)

# 加载图片
image = cv2.imread("product.jpg")

# 检测 logo
results = detector.detect(image)

# 打印结果
for result in results:
    print(f"品牌: {result.details['brand']}")
    print(f"置信度: {result.confidence:.2%}")
```

## 🔗 相关资源

- LogoDet-3K Dataset: https://github.com/Wangjing1551/LogoDet-3K-Dataset
- Roboflow Universe: https://universe.roboflow.com/
- YOLOv8 文档: https://docs.ultralytics.com/
