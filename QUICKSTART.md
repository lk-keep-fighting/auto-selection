# 快速开始指南

## 5 分钟快速上手

### 1️⃣ 安装依赖

```bash
# 安装 Python 依赖
pip install -e .

# 或者手动安装
pip install selenium requests numpy opencv-python-headless Pillow
```

### 2️⃣ 安装 ChromeDriver

**macOS:**
```bash
brew install chromedriver
```

**其他系统:**
访问 https://chromedriver.chromium.org/downloads 下载对应版本

### 3️⃣ 检查环境

```bash
python3 check_environment.py
```

看到 "🎉 所有检查通过" 即可继续。

### 4️⃣ 运行程序

```bash
python3 browser_automation.py
```

或使用快捷脚本：

```bash
./run_browser_automation.sh
```

### 5️⃣ 操作流程

1. **浏览器自动打开** - 等待页面加载
2. **手动登录** - 如果需要的话，完成登录
3. **按 Enter 继续** - 在终端按回车键
4. **自动处理** - 程序自动获取图片并识别
5. **查看结果** - 打开 `browser_analysis_results.json`

## 配置说明

编辑 `config.json` 文件：

```json
{
  "targetUrl": "目标网址",
  "selectors": {
    "imageList": ".图片列表容器",
    "cardItem": ".商品卡片",
    "imageItem": "img.商品图片"
  }
}
```

## 输出文件

- `browser_analysis_results.json` - 分析结果
- `downloaded_images/` - 下载的图片（如果启用）
- `browser_cookies.json` - 保存的登录状态

## 常用命令

```bash
# 环境检查
python3 check_environment.py

# 运行程序（默认配置）
python3 browser_automation.py

# 使用自定义配置
python3 browser_automation.py my_config.json

# 查看帮助
python3 browser_automation.py --help
```

## 下一步

- 📖 阅读详细文档：[BROWSER_AUTOMATION_GUIDE.md](BROWSER_AUTOMATION_GUIDE.md)
- 🔧 修改配置适配其他网站：参考 [config.template.json](config.template.json)
- 🤖 启用 YOLO 检测：`pip install ultralytics`

## 故障排除

### ❌ ChromeDriver 版本不匹配

```bash
# 检查 Chrome 版本
google-chrome --version  # Linux
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version  # macOS

# 下载匹配版本的 ChromeDriver
# https://chromedriver.chromium.org/downloads
```

### ❌ 无法找到元素

检查 `config.json` 中的 CSS 选择器是否正确：

1. 打开浏览器开发者工具（F12）
2. 使用元素选择器找到目标元素
3. 复制 CSS 选择器
4. 更新配置文件

### ❌ 图片下载失败

可能原因：
- 网络问题
- 需要登录后才能访问图片
- 图片 URL 格式不正确

解决方法：确保已登录并保存了登录状态。

## 支持

遇到问题？

1. 运行环境检查：`python3 check_environment.py`
2. 查看详细文档：`BROWSER_AUTOMATION_GUIDE.md`
3. 检查日志输出

---

✨ **开始使用浏览器自动化图片识别吧！**
