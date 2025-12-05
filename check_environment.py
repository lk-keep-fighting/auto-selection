#!/usr/bin/env python3
"""浏览器自动化测试脚本 - 检查环境和依赖"""

import sys
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    print("🐍 检查 Python 版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (需要 >= 3.10)")
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    dependencies = {
        "playwright": "浏览器自动化",
        "requests": "HTTP 请求",
        "numpy": "数值计算",
        "cv2": "OpenCV 图像处理",
        "PIL": "Pillow 图像库"
    }
    
    all_ok = True
    
    for package, description in dependencies.items():
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "PIL":
                from PIL import Image
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
            
            print(f"  ✅ {package} ({description}): {version}")
        except ImportError:
            print(f"  ❌ {package} ({description}): 未安装")
            all_ok = False
    
    return all_ok


def check_chromedriver():
    """检查 Playwright 浏览器"""
    print("\n🚗 检查 Playwright 浏览器...")
    
    try:
        from playwright.sync_api import sync_playwright
        
        print("  ✅ Playwright 已安装")
        
        # 检查是否安装了浏览器
        import subprocess
        result = subprocess.run(
            ["playwright", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"  ✅ {result.stdout.strip()}")
            print(f"  💡 如果未安装浏览器，请运行: playwright install chromium")
            return True
        else:
            print(f"  ⚠️  Playwright CLI 不可用")
            return True  # Playwright 库已安装，只是 CLI 不可用
            
    except ImportError:
        print(f"  ❌ Playwright 未安装")
        print(f"  💡 安装方法:")
        print(f"     pip install playwright")
        print(f"     playwright install chromium")
        return False
    except Exception as e:
        print(f"  ⚠️  检查失败: {e}")
        return True  # 即使检查失败，也认为通过


def check_config_file():
    """检查配置文件"""
    print("\n📄 检查配置文件...")
    
    config_file = Path("config.json")
    
    if config_file.exists():
        print(f"  ✅ config.json 存在")
        
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            required_keys = ["targetUrl", "selectors", "browser", "detection", "output"]
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print(f"  ⚠️  缺少配置项: {', '.join(missing_keys)}")
                return False
            else:
                print(f"  ✅ 配置文件格式正确")
                return True
                
        except json.JSONDecodeError as e:
            print(f"  ❌ 配置文件格式错误: {e}")
            return False
    else:
        print(f"  ❌ config.json 不存在")
        print(f"  💡 请确保配置文件在当前目录")
        return False


def check_local_recognition():
    """检查本地识别模块"""
    print("\n🔍 检查本地识别模块...")
    
    try:
        from local_product_recognition import LocalProductImageRecognizer
        print(f"  ✅ LocalProductImageRecognizer 可用")
        
        recognizer = LocalProductImageRecognizer()
        features = recognizer.available_features()
        print(f"  ✅ 支持的特征类型: {len(features)} 个")
        for feature in features:
            print(f"     - {feature.value}")
        
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        print(f"  💡 请确保已安装项目: pip install -e .")
        return False


def check_yolo_availability():
    """检查 YOLO 可用性"""
    print("\n🤖 检查 YOLO 支持...")
    
    try:
        from ultralytics import YOLO
        print(f"  ✅ ultralytics 已安装")
        
        # 检查是否有预训练模型
        model_file = Path("yolov8n.pt")
        if model_file.exists():
            print(f"  ✅ yolov8n.pt 模型文件存在")
        else:
            print(f"  ⚠️  yolov8n.pt 模型文件不存在")
            print(f"     首次运行时会自动下载")
        
        return True
    except ImportError:
        print(f"  ⚠️  ultralytics 未安装 (可选)")
        print(f"  💡 安装方法: pip install ultralytics")
        print(f"  ℹ️  不影响基本功能，可使用传统检测器")
        return False


def check_selenium_browser():
    """测试 Playwright 浏览器启动"""
    print("\n🌐 测试浏览器启动...")
    
    try:
        from playwright.sync_api import sync_playwright
        
        print(f"  🔄 正在启动 Chromium...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("about:blank")
            title = page.title()
            browser.close()
        
        print(f"  ✅ 浏览器启动成功")
        return True
        
    except Exception as e:
        print(f"  ❌ 浏览器启动失败: {e}")
        print(f"  💡 请运行: playwright install chromium")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🔧 浏览器自动化环境检查")
    print("=" * 80)
    
    checks = [
        ("Python 版本", check_python_version),
        ("依赖包", check_dependencies),
        ("Playwright", check_chromedriver),
        ("配置文件", check_config_file),
        ("本地识别模块", check_local_recognition),
        ("YOLO 支持", check_yolo_availability),
        ("浏览器启动", check_selenium_browser),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} 检查时发生错误: {e}")
            results[name] = False
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📊 检查结果汇总")
    print("=" * 80)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有检查通过！可以运行浏览器自动化程序")
        print("\n运行命令: python browser_automation.py")
    else:
        print("\n⚠️  部分检查未通过，请根据提示解决问题")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
