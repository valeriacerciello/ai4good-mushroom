#!/bin/bash

echo "🍄 Setting up CLIP-based Mushroom Classification System"
echo "======================================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv mushroom_clip_env
    source mushroom_clip_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install PyTorch
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CLIP
echo "🖼️ Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Test installation
echo "🧪 Testing CLIP installation..."
python3 -c "import clip; print('✅ CLIP installed successfully')"
python3 -c "import torch; print('✅ PyTorch installed successfully')"
python3 -c "import pandas; print('✅ Pandas installed successfully')"

echo ""
echo "🎉 Setup complete! You can now:"
echo "1. Run: python generate_clip_prompts.py  (already done)"
echo "2. Run: python clip_mushroom_classifier.py --demo"
echo "3. Run: python demo_prompts.py  (to see prompt analysis)"

echo ""
echo "📊 Your mushroom dataset:"
echo "- Location: /zfs/ai4good/datasets/mushroom/"
echo "- Species: 169 unique fungi species"  
echo "- Images: ~720,000 total"
echo "- Prompts: Generated for all 169 species ✅"

if [ "$create_venv" = "y" ]; then
    echo ""
    echo "💡 To activate virtual environment in future sessions:"
    echo "source ~/mushroom_clip/mushroom_clip_env/bin/activate"
fi

echo ""
echo "🚀 Ready to classify mushrooms with CLIP!"