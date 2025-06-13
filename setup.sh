#!/bin/bash
# OpenHands LLM API Aggregator Setup Script

set -e

echo "🚀 Setting up OpenHands LLM API Aggregator..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Optional: Install ML features
read -p "🤖 Install ML features (PyTorch)? [y/N]: " install_ml
if [[ $install_ml =~ ^[Yy]$ ]]; then
    echo "📦 Installing ML dependencies..."
    pip install -r requirements-ml.txt
fi

# Create data directory
mkdir -p data
echo "✅ Created data directory"

# Check .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please copy .env.example to .env and configure it."
    cp .env.example .env
    echo "📝 Created .env file from template"
    echo "🚨 IMPORTANT: Edit .env file and change the default tokens!"
else
    echo "✅ .env file exists"
fi

# Run basic tests
echo "🧪 Running basic tests..."
python test_simple.py

echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit .env file and change default tokens"
echo "   2. Run: python setup.py configure (to add API credentials)"
echo "   3. Start server: python main.py"
echo "   4. Access web UI: http://localhost:8501"
echo ""
echo "🐳 Docker option:"
echo "   docker-compose up -d"
