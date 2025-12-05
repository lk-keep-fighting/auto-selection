#!/bin/bash
# 浏览器自动化快速启动脚本

echo "🤖 浏览器自动化图片识别系统"
echo "================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装"
    exit 1
fi

echo "✅ Python: $(python3 --version)"

# 检查依赖
echo ""
echo "📦 检查依赖..."
python3 check_environment.py

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  环境检查发现问题，是否继续？(y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 运行程序
echo ""
echo "🚀 启动浏览器自动化程序..."
echo ""
python3 browser_automation.py "$@"
