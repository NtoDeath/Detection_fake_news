#!/bin/bash
# Setup script for Fake News Detection CLI Tool
# Run: bash setup.sh (from within cli_tool/ or from parent directory)

set -e

echo "🔍 Fake News Detection CLI - Setup"
echo "===================================="
echo ""

# 1. Check Python
echo "1️⃣  Checking Python..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found! Please install Python 3.9+"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION found"
echo ""

# 2. Install dependencies
echo "2️⃣  Installing dependencies..."
echo "(Showing real-time progress - this may take 1-2 minutes)"
echo ""
$PYTHON_CMD -m pip install -r requirements.txt --progress-bar on
echo ""
echo "✅ Dependencies installed"
echo ""

# 3. Copy models if needed
echo "3️⃣  Checking models..."
if [ -d "models" ]; then
    echo "✅ Models directory found"
else
    echo "⚠️  Models not found, trying to copy..."
    if [ -f "models_copy.py" ]; then
        $PYTHON_CMD models_copy.py
    else
        echo "⚠️  models_copy.py not found. Using fallback..."
        mkdir -p models
    fi
fi
echo ""

# 4. Verify setup
echo "4️⃣  Verifying setup..."
$PYTHON_CMD -c "
try:
    from main import app
    print('✅ CLI imports successful')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
" || exit 1
echo ""

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick start:"
echo "   Interactive REPL mode:"
echo "   $ python main.py"
echo ""
echo "   Single prediction:"
echo "   $ python main.py predict \"Your text here\""
echo ""
echo "   Model info:"
echo "   $ python main.py info"
echo ""
echo "For more details, see: STANDALONE_README.md"
