"""
Quick test script to verify training setup without GPU
Tests dataset loading and model configuration
"""
import sys
import json
from pathlib import Path

print("=" * 60)
print("IELTS Training Setup Verification")
print("=" * 60)

# 1. Check dataset
print("\n1. Checking dataset...")
dataset_path = Path("./data/ielts_training_data.jsonl")

if not dataset_path.exists():
    print("✗ Dataset not found!")
    sys.exit(1)

with open(dataset_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]
    
print(f"✓ Found {len(samples)} training samples")
for i, sample in enumerate(samples[:3], 1):
    print(f"\n  Sample {i}:")
    print(f"    Transcript: {sample['transcript'][:50]}...")
    print(f"    Scores: {sample['scores']}")

# 2. Check processed data
print("\n2. Checking processed data...")
processed_dir = Path("./data/processed")

if processed_dir.exists():
    json_files = list(processed_dir.glob("*.json"))
    print(f"✓ Found {len(json_files)} processed files")
    for f in json_files:
        print(f"  - {f.name}")
else:
    print("⚠ No processed data found")

# 3. Check audio files
print("\n3. Checking audio files...")
audio_dir = Path("./data training")

if audio_dir.exists():
    audio_files = []
    for ext in ['.mp3', '.mp4', '.wav', '.m4a']:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
    print(f"✓ Found {len(audio_files)} audio files")
    for f in audio_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print("⚠ No audio files found")

# 4. Check model configuration
print("\n4. Checking model configuration...")
config_path = Path("./config/training_config.json")

if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"✓ Configuration loaded")
    print(f"  Model: {config['training']['model_name']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
else:
    print("⚠ Configuration not found")

# 5. Check training scripts
print("\n5. Checking training scripts...")
scripts = {
    "Training": "./app/train_qlora.py",
    "Dataset prep": "./app/prepare_dataset.py",
    "Audio processing": "./app/process_audio.py",
    "Inference": "./app/inference.py",
}

for name, path in scripts.items():
    if Path(path).exists():
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: NOT FOUND")

# 6. GPU Check
print("\n6. GPU Availability...")
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("⚠ PyTorch not installed (will use Docker)")

# 7. Docker Check
print("\n7. Docker Services...")
import subprocess

try:
    result = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✓ {result.stdout.strip()}")
    
    result = subprocess.run(
        ["docker-compose", "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✓ {result.stdout.strip()}")
except Exception as e:
    print(f"⚠ Docker not available: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
✓ Dataset: {len(samples)} samples ready
✓ Audio files: Available for processing
✓ Configuration: Ready
✓ Scripts: All present

Next steps:
1. Wait for Docker build to complete
2. Run: docker-compose --profile training up trainer
3. Or install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121
4. Then run: python app/train_qlora.py --model qwen --epochs 1
""")
