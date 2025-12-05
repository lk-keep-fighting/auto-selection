# 浏览器自动化图片识别系统使用指南

## 功能概述

本程序提供浏览器自动化功能，可以：
1. 自动打开浏览器并访问指定网址
2. 等待用户登录后保存登录状态
3. 自动获取页面中的商品图片列表
4. 使用现有的图像识别功能检测敏感特征
5. 生成分析报告

## 安装依赖

首先安装必要的 Python 依赖：

```bash
pip install -e .
```

或者手动安装：

```bash
pip install selenium requests numpy opencv-python-headless Pillow
```

## 安装 ChromeDriver

浏览器自动化需要 ChromeDriver，请按以下方式安装：

### macOS
```bash
brew install chromedriver
```

### 手动安装
1. 访问 https://chromedriver.chromium.org/downloads
2. 下载与你的 Chrome 浏览器版本匹配的 ChromeDriver
3. 解压并将 chromedriver 放到 PATH 路径中

## 配置文件说明

程序使用 `config.json` 配置文件，主要配置项：

```json
{
  "targetUrl": "目标网址",
  "selectors": {
    "imageList": "图片列表容器选择器",
    "cardItem": "商品卡片选择器",
    "imageItem": "图片元素选择器",
    "asinSelector": "ASIN 选择器",
    "titleSelector": "标题选择器"
  },
  "browser": {
    "headless": false,           // 是否无头模式
    "window_size": "1920,1080",  // 窗口大小
    "user_data_dir": "./browser_profile"  // 用户数据目录（保存登录状态）
  },
  "detection": {
    "use_yolo": true,             // 使用 YOLO 检测
    "confidence_threshold": 0.5,  // 置信度阈值
    "enable_logo_detection": false,
    "logo_model_path": null
  },
  "output": {
    "results_file": "browser_analysis_results.json",
    "images_folder": "downloaded_images",
    "save_images": true  // 是否保存图片到本地
  }
}
```

## 使用方法

### 基本用法

运行浏览器自动化程序：

```bash
python browser_automation.py
```

### 使用自定义配置文件

```bash
python browser_automation.py my_config.json
```

## 工作流程

1. **启动浏览器**: 程序会自动打开 Chrome 浏览器
2. **访问网址**: 自动跳转到配置的目标网址
3. **等待登录**: 程序会暂停，等待你手动完成登录操作
4. **保存状态**: 按 Enter 后，程序会保存登录状态（Cookie）到 `browser_cookies.json`
5. **获取图片**: 自动滚动页面并提取所有商品图片
6. **图像识别**: 使用 YOLO 或传统方法检测敏感特征
7. **保存结果**: 将分析结果保存到 JSON 文件

## 输出说明

### 图片文件
如果启用了 `save_images`，下载的图片会保存到 `downloaded_images/` 目录，命名格式：
```
001_B08XYZ1234.jpg
002_B07ABC5678.jpg
...
```

### 分析结果
结果保存在 `browser_analysis_results.json`，包含：

```json
{
  "passed": [
    {
      "asin": "B08XYZ1234",
      "title": "商品标题",
      "image_url": "图片URL",
      "image_file": "001_B08XYZ1234.jpg"
    }
  ],
  "detected": [
    {
      "asin": "B07ABC5678",
      "title": "商品标题",
      "image_url": "图片URL",
      "image_file": "002_B07ABC5678.jpg",
      "features": [
        {
          "feature": "person",
          "confidence": 0.85
        }
      ]
    }
  ],
  "summary": {
    "total": 60,
    "passed_count": 45,
    "detected_count": 15,
    "confidence_threshold": 0.5
  }
}
```

## 检测的敏感特征

- **person**: 人物
- **brand_logo**: 品牌 Logo
- **chemicals**: 危化品标识
- **electronics**: 电子设备
- **controlled_props**: 受控物品（刀具等）
- **toys**: 玩具/儿童用品

## 常见问题

### Q: ChromeDriver 版本不匹配？
A: 确保 ChromeDriver 版本与你的 Chrome 浏览器版本匹配

### Q: 无法加载图片？
A: 检查网络连接，或调整 `_scroll_page()` 的滚动次数

### Q: 登录状态丢失？
A: 使用 `user_data_dir` 配置项来持久化浏览器数据

### Q: 如何使用 YOLO 检测？
A: 首先安装 ultralytics：
```bash
pip install ultralytics
```

然后在配置中设置 `"use_yolo": true`

## 进阶用法

### 仅获取图片，不进行识别

修改代码，注释掉 `analyze_images` 调用：

```python
# results = self.analyze_images(images_info)
```

### 使用已保存的登录状态

取消注释 `run()` 方法中的这一行：

```python
self.load_login_state()
```

### 修改选择器适配其他网站

根据目标网站的 HTML 结构，修改 `config.json` 中的 `selectors` 配置。

## 示例：SellerSprite 网站

默认配置已经适配了 SellerSprite 网站的商品研究页面。运行程序后：

1. 浏览器会自动打开并跳转到商品列表页
2. 等待登录并筛选条件加载完成
3. 按 Enter 继续
4. 程序会自动提取所有商品图片并进行识别

## 许可证

MIT License
