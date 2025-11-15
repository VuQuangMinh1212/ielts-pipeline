"""
Quick GPU check script for Docker container
"""
import torch
import sys

print("=" * 60)
print("🔍 GPU Configuration Check")
print("=" * 60)

print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test CUDA
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print("✓ CUDA computation test: PASSED")
    except Exception as e:
        print(f"✗ CUDA computation test: FAILED - {e}")
        sys.exit(1)
else:
    print("⚠️ CUDA not available - will use CPU (very slow)")
    
print("\n" + "=" * 60)
print("✅ GPU check complete!")
print("=" * 60)
