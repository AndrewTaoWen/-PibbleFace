#!/bin/bash
# Activation script for Face Emoji Matcher virtual environment

echo "🎭 Activating Face Emoji Matcher virtual environment..."

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated!"
    echo ""
    echo "To run the application:"
    echo "  Web interface: python app.py"
    echo "  Command line:  python simple_demo.py"
    echo ""
    echo "To deactivate: deactivate"
else
    echo "❌ Virtual environment not found!"
    echo "Run 'python3 setup.py' first to create it."
fi
