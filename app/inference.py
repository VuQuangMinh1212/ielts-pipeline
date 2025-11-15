"""
Inference server using vLLM for efficient model serving
Supports CPU mode and quantized models for RTX 3050 (4GB)
"""
import os
import json
from typing import Optional, Dict, List
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠ vLLM not available. Install with: pip install vllm")


class VLLMInferenceServer:
    """vLLM-based inference server for IELTS models"""
    
    def __init__(
        self,
        model_path: str = "./models/ielts-finetuned",
        gpu_memory_utilization: float = 0.9,
        quantization: Optional[str] = "awq",  # or "gptq", "squeezellm"
        max_model_len: int = 512,
        cpu_mode: bool = False,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required. Install with: pip install vllm")
            
        self.model_path = model_path
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization if not cpu_mode else None,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=cpu_mode,  # CPU mode
        )
        
        print(f"✓ Loaded model from {model_path}")
        
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """Generate responses for given prompts"""
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
        
    def score_transcript(self, transcript: str) -> Dict:
        """Score an IELTS transcript"""
        
        prompt = f"""### Instruction:
Score the following IELTS speaking response. Provide scores for Fluency, Lexical Resource, Grammatical Range, and Pronunciation.

### Transcript:
{transcript}

### Response:
"""
        
        response = self.generate([prompt])[0]
        
        # Parse JSON response
        try:
            scores = json.loads(response)
            return scores
        except json.JSONDecodeError:
            # Fallback parsing
            return {"raw_response": response}


class LlamaCppInferenceServer:
    """llama.cpp-based inference for CPU deployment"""
    
    def __init__(
        self,
        model_path: str = "./models/ielts-finetuned/model.gguf",
        n_ctx: int = 512,
        n_threads: int = 4,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required. Install with: pip install llama-cpp-python")
            
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only
        )
        
        print(f"✓ Loaded model from {model_path} (llama.cpp)")
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate response for prompt"""
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        
        return output["choices"][0]["text"]
        
    def score_transcript(self, transcript: str) -> Dict:
        """Score an IELTS transcript"""
        
        prompt = f"""### Instruction:
Score the following IELTS speaking response. Provide scores for Fluency, Lexical Resource, Grammatical Range, and Pronunciation.

### Transcript:
{transcript}

### Response:
"""
        
        response = self.generate(prompt)
        
        try:
            scores = json.loads(response)
            return scores
        except json.JSONDecodeError:
            return {"raw_response": response}


def convert_to_gguf(
    model_path: str = "./models/ielts-finetuned",
    output_path: str = "./models/ielts-finetuned/model.gguf",
    quantization: str = "Q4_K_M",
):
    """
    Convert model to GGUF format for llama.cpp
    
    Quantization options:
    - Q4_K_M: 4-bit quantization (recommended for 4GB GPU)
    - Q5_K_M: 5-bit quantization
    - Q8_0: 8-bit quantization
    """
    import subprocess
    
    print(f"Converting {model_path} to GGUF format...")
    
    # Convert to GGUF
    cmd = [
        "python",
        "-m",
        "llama_cpp.convert",
        model_path,
        "--outfile",
        output_path,
        "--outtype",
        quantization,
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Model converted to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e}")
        

class AutotrainInferenceServer:
    """Autotrain-based inference wrapper"""
    
    def __init__(self, model_path: str = "./models/ielts-finetuned"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers required")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        print(f"✓ Loaded model from {model_path} (Autotrain)")
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate response"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]  # Remove prompt
        
        return response
        
    def score_transcript(self, transcript: str) -> Dict:
        """Score transcript"""
        prompt = f"""### Instruction:
Score the following IELTS speaking response. Provide scores for Fluency, Lexical Resource, Grammatical Range, and Pronunciation.

### Transcript:
{transcript}

### Response:
"""
        
        response = self.generate(prompt)
        
        try:
            scores = json.loads(response)
            return scores
        except json.JSONDecodeError:
            return {"raw_response": response}


def main():
    """Test inference servers"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IELTS Model Inference Server")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "llamacpp", "autotrain"],
        help="Inference backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ielts-finetuned",
        help="Model path",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU mode",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        help="Test transcript to score",
    )
    
    args = parser.parse_args()
    
    # Initialize server
    if args.backend == "vllm":
        server = VLLMInferenceServer(
            model_path=args.model,
            cpu_mode=args.cpu,
        )
    elif args.backend == "llamacpp":
        server = LlamaCppInferenceServer(model_path=args.model)
    else:
        server = AutotrainInferenceServer(model_path=args.model)
        
    # Test with sample or provided transcript
    test_transcript = args.transcript or "Well, I think technology has changed our lives significantly."
    
    print(f"\n📝 Testing with transcript: {test_transcript}\n")
    scores = server.score_transcript(test_transcript)
    print(f"✓ Scores: {json.dumps(scores, indent=2)}")


if __name__ == "__main__":
    main()
