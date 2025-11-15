@echo off
REM Quick start script for Windows - Training and Deployment

echo.
echo 🚀 IELTS Model Training ^& Deployment Setup
echo ==========================================

REM Check Python version
python --version
echo.

echo Select operation:
echo 1) Install dependencies
echo 2) Prepare training dataset
echo 3) Train model (Qwen 1.5B)
echo 4) Train model (Phi-2)
echo 5) Train model (Gemma 2B)
echo 6) Convert to GGUF (llama.cpp)
echo 7) Start inference server (vLLM)
echo 8) Start inference server (CPU - llama.cpp)
echo 9) Docker training setup
echo.

set /p choice="Enter choice [1-9]: "

if "%choice%"=="1" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo ✓ Installation complete!
) else if "%choice%"=="2" (
    echo Preparing training dataset...
    python app/prepare_dataset.py --validate
    echo ✓ Dataset prepared!
) else if "%choice%"=="3" (
    echo Training with Qwen 1.5B...
    python app/train_qlora.py --model qwen --epochs 3 --batch-size 1
) else if "%choice%"=="4" (
    echo Training with Phi-2...
    python app/train_qlora.py --model phi2 --epochs 3 --batch-size 1
) else if "%choice%"=="5" (
    echo Training with Gemma 2B...
    python app/train_qlora.py --model gemma --epochs 3 --batch-size 1
) else if "%choice%"=="6" (
    set /p model_path="Model path [./models/ielts-finetuned]: "
    if "%model_path%"=="" set model_path=./models/ielts-finetuned
    echo Converting model to GGUF...
    python app/convert_model.py --model %model_path% --format gguf --quantization Q4_K_M
) else if "%choice%"=="7" (
    set /p model_path="Model path [./models/ielts-finetuned]: "
    if "%model_path%"=="" set model_path=./models/ielts-finetuned
    echo Starting vLLM inference server...
    python app/inference.py --backend vllm --model %model_path%
) else if "%choice%"=="8" (
    set /p model_path="Model path [./models/ielts-finetuned/model.gguf]: "
    if "%model_path%"=="" set model_path=./models/ielts-finetuned/model.gguf
    echo Starting CPU inference server...
    python app/inference.py --backend llamacpp --model %model_path% --cpu
) else if "%choice%"=="9" (
    echo Starting Docker training...
    docker-compose --profile training up trainer
) else (
    echo Invalid choice
    exit /b 1
)

echo.
echo ✓ Operation complete!
pause
