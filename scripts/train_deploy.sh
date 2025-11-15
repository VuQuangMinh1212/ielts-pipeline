#!/bin/bash
# Quick start script for training and deployment

set -e

echo "🚀 IELTS Model Training & Deployment Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('✓ CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "⚠ PyTorch not installed yet"

echo ""
echo "Select operation:"
echo "1) Install dependencies"
echo "2) Prepare training dataset"
echo "3) Train model (Qwen 1.5B)"
echo "4) Train model (Phi-2)"
echo "5) Train model (Gemma 2B)"
echo "6) Convert to GGUF (llama.cpp)"
echo "7) Start inference server (vLLM)"
echo "8) Start inference server (CPU - llama.cpp)"
echo "9) Docker training setup"

read -p "Enter choice [1-9]: " choice

case $choice in
    1)
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo "✓ Installation complete!"
        ;;
    2)
        echo "Preparing training dataset..."
        python3 app/prepare_dataset.py --validate
        echo "✓ Dataset prepared!"
        ;;
    3)
        echo "Training with Qwen 1.5B..."
        python3 app/train_qlora.py --model qwen --epochs 3 --batch-size 1
        ;;
    4)
        echo "Training with Phi-2..."
        python3 app/train_qlora.py --model phi2 --epochs 3 --batch-size 1
        ;;
    5)
        echo "Training with Gemma 2B..."
        python3 app/train_qlora.py --model gemma --epochs 3 --batch-size 1
        ;;
    6)
        echo "Converting model to GGUF..."
        read -p "Model path [./models/ielts-finetuned]: " model_path
        model_path=${model_path:-./models/ielts-finetuned}
        python3 app/convert_model.py --model $model_path --format gguf --quantization Q4_K_M
        ;;
    7)
        echo "Starting vLLM inference server..."
        read -p "Model path [./models/ielts-finetuned]: " model_path
        model_path=${model_path:-./models/ielts-finetuned}
        python3 app/inference.py --backend vllm --model $model_path
        ;;
    8)
        echo "Starting CPU inference server..."
        read -p "Model path [./models/ielts-finetuned/model.gguf]: " model_path
        model_path=${model_path:-./models/ielts-finetuned/model.gguf}
        python3 app/inference.py --backend llamacpp --model $model_path --cpu
        ;;
    9)
        echo "Starting Docker training..."
        docker-compose --profile training up trainer
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Operation complete!"
