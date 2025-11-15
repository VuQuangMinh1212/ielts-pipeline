"""
Simple training test without full dependencies
Tests dataset preparation and config loading
"""
import json
import sys
from pathlib import Path

print("=" * 60)
print("IELTS Training - Quick Test")
print("=" * 60)

# Test 1: Check dataset
print("\n1. Checking training dataset...")
dataset_path = Path("./data/ielts_training_data.jsonl")

if dataset_path.exists():
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    print(f"✓ Found {len(samples)} training samples")
    
    if samples:
        print(f"\nSample transcript:")
        print(f"  {samples[0]['transcript'][:100]}...")
        print(f"  Scores: {samples[0]['scores']}")
else:
    print("✗ No dataset found. Creating sample dataset...")
    sys.path.append(".")
    from app.prepare_dataset import prepare_dataset
    prepare_dataset()
    print("✓ Dataset created!")

# Test 2: Check config
print("\n2. Checking training config...")
config_path = Path("./config/training_config.json")

if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"✓ Config loaded")
    print(f"  Model: {config['training']['model_name']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
else:
    print("✗ Config not found!")

# Test 3: Check CUDA (if PyTorch available)
print("\n3. Checking CUDA availability...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU with simple operation
        x = torch.randn(100, 100).cuda()
        y = x @ x.T
        print(f"✓ GPU computation test passed!")
    else:
        print("⚠ CUDA not available - will use CPU (slow)")
        
except ImportError:
    print("⚠ PyTorch not installed")
    print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")

# Test 4: Check model output directory
print("\n4. Checking model output directory...")
model_dir = Path("./models")
model_dir.mkdir(exist_ok=True)
print(f"✓ Model directory: {model_dir.absolute()}")

# Test 5: Check dependencies
print("\n5. Checking key dependencies...")
dependencies = {
    "transformers": "Hugging Face Transformers",
    "peft": "PEFT (LoRA)",
    "trl": "TRL (Training)",
    "datasets": "Datasets",
    "bitsandbytes": "BitsAndBytes (Quantization)",
    "accelerate": "Accelerate",
}

missing = []
for pkg, name in dependencies.items():
    try:
        __import__(pkg)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - NOT INSTALLED")
        missing.append(pkg)

if missing:
    print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
    print("Install with:")
    print(f"  pip install {' '.join(missing)}")
else:
    print("\n✓ All dependencies installed!")

# Summary
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

if dataset_path.exists() and config_path.exists():
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ Ready to train with GPU!")
            print("\nStart training with:")
            print("  python app/train_qlora.py --model qwen --epochs 3")
        else:
            print("⚠ Ready to train but no GPU detected")
            print("  Training will be very slow without GPU")
    except ImportError:
        print("✗ Install PyTorch first:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
else:
    print("✗ Setup incomplete - check errors above")

print("=" * 60)
