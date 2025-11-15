"""
Model quantization utilities for deployment
Supports GGUF, AWQ, GPTQ conversion
"""
import os
import subprocess
from pathlib import Path
from typing import Optional


def convert_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quantization: str = "Q4_K_M",
):
    """
    Convert HuggingFace model to GGUF format for llama.cpp
    
    Args:
        model_path: Path to the fine-tuned model
        output_path: Output GGUF file path
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0)
    """
    if output_path is None:
        output_path = os.path.join(model_path, "model.gguf")
        
    print(f"Converting {model_path} to GGUF ({quantization})...")
    
    # First convert to GGUF format
    convert_cmd = f"python -m llama_cpp.convert {model_path}"
    
    # Then quantize
    quantize_cmd = [
        "llama-quantize",
        os.path.join(model_path, "ggml-model-f16.gguf"),
        output_path,
        quantization,
    ]
    
    try:
        # Note: Requires llama.cpp to be installed
        print("Step 1: Converting to GGUF format...")
        os.system(convert_cmd)
        
        print(f"Step 2: Quantizing to {quantization}...")
        subprocess.run(quantize_cmd, check=True)
        
        print(f"✓ Model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print("\nMake sure llama.cpp is installed:")
        print("git clone https://github.com/ggerganov/llama.cpp")
        print("cd llama.cpp && make")
        return None


def convert_to_awq(
    model_path: str,
    output_path: Optional[str] = None,
    bits: int = 4,
):
    """
    Convert model to AWQ format (4-bit quantization)
    Optimized for fast inference
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("✗ AutoAWQ not installed. Install with: pip install autoawq")
        return None
        
    if output_path is None:
        output_path = model_path + "-awq"
        
    print(f"Converting {model_path} to AWQ {bits}-bit...")
    
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Quantize
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": bits,
    }
    
    model.quantize(tokenizer, quant_config=quant_config)
    
    # Save
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ AWQ model saved to {output_path}")
    return output_path


def convert_to_gptq(
    model_path: str,
    output_path: Optional[str] = None,
    bits: int = 4,
):
    """
    Convert model to GPTQ format (4-bit quantization)
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("✗ AutoGPTQ not installed. Install with: pip install auto-gptq")
        return None
        
    if output_path is None:
        output_path = model_path + "-gptq"
        
    print(f"Converting {model_path} to GPTQ {bits}-bit...")
    
    # Quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,
    )
    
    # Load and quantize
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Save
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ GPTQ model saved to {output_path}")
    return output_path


def main():
    """CLI for model conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert models for deployment")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model to convert",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["gguf", "awq", "gptq"],
        required=True,
        help="Target format",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bits (4, 8)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        help="GGUF quantization type",
    )
    
    args = parser.parse_args()
    
    if args.format == "gguf":
        convert_to_gguf(args.model, args.output, args.quantization)
    elif args.format == "awq":
        convert_to_awq(args.model, args.output, args.bits)
    elif args.format == "gptq":
        convert_to_gptq(args.model, args.output, args.bits)


if __name__ == "__main__":
    main()
