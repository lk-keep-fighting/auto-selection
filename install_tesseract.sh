#!/bin/bash
echo "=========================================="
echo "🚀 安装 Tesseract OCR"
echo "=========================================="

# 检查 Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew 未安装"
    echo "请先安装 Homebrew: https://brew.sh"
    exit 1
fi

# 安装 Tesseract
echo ""
echo "1️⃣ 安装 Tesseract OCR 引擎..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract 已安装: $(tesseract --version | head -n1)"
else
    echo "📦 正在安装 Tesseract..."
    brew install tesseract
    if [ $? -eq 0 ]; then
        echo "✅ Tesseract 安装成功"
    else
        echo "❌ Tesseract 安装失败"
        exit 1
    fi
fi

# 安装 Python 绑定
echo ""
echo "2️⃣ 安装 Python 绑定..."
pip3 install pytesseract

# 验证安装
echo ""
echo "3️⃣ 验证安装..."
python3 << 'PYTHON'
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"✅ 安装成功! Tesseract 版本: {version}")
    print("\n现在可以运行程序了:")
    print("  python3 browser_automation.py")
except Exception as e:
    print(f"❌ 验证失败: {e}")
    exit(1)
PYTHON

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
