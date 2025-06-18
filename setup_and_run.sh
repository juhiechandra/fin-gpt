#!/bin/bash

echo "🚀 FinGPT Forecaster Setup Script"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set environment variable for token
export HUGGING_FACE_TOKEN="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"

echo "✅ Setup complete!"
echo ""
echo "🎯 Running FinGPT Forecaster..."
echo "=================================="

# Run the FinGPT forecaster
python3 fingpt_forecaster_setup.py 