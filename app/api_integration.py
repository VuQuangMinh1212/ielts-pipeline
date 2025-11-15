"""
Integration example: Using fine-tuned model in the main API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os

# Import inference server
from .inference import VLLMInferenceServer, LlamaCppInferenceServer

router = APIRouter()


class TranscriptScoreRequest(BaseModel):
    transcript: str
    use_local_model: bool = False  # Toggle between Gemini and local model


class TranscriptScoreResponse(BaseModel):
    scores: Dict[str, float]
    backend: str
    model_path: Optional[str] = None


# Initialize local model (lazy loading)
_inference_server = None


def get_inference_server():
    """Lazy load inference server"""
    global _inference_server
    
    if _inference_server is None:
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/ielts-finetuned")
        backend = os.getenv("INFERENCE_BACKEND", "vllm")
        
        try:
            if backend == "vllm":
                _inference_server = VLLMInferenceServer(
                    model_path=model_path,
                    gpu_memory_utilization=0.9,
                )
            else:  # llamacpp
                _inference_server = LlamaCppInferenceServer(
                    model_path=os.path.join(model_path, "model.gguf")
                )
            print(f"✓ Loaded local model: {model_path}")
        except Exception as e:
            print(f"⚠ Failed to load local model: {e}")
            _inference_server = None
            
    return _inference_server


@router.post("/api/v1/score-transcript", response_model=TranscriptScoreResponse)
async def score_transcript(request: TranscriptScoreRequest):
    """
    Score IELTS transcript using either:
    1. Fine-tuned local model (QLoRA)
    2. Google Gemini API (fallback)
    """
    
    if request.use_local_model:
        # Use fine-tuned local model
        server = get_inference_server()
        
        if server is None:
            raise HTTPException(
                status_code=503,
                detail="Local model not available. Please use Gemini API."
            )
            
        try:
            scores = server.score_transcript(request.transcript)
            
            return TranscriptScoreResponse(
                scores=scores,
                backend="local",
                model_path=os.getenv("LOCAL_MODEL_PATH", "./models/ielts-finetuned")
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
    else:
        # Use existing Gemini API
        from .genai_client import client
        import json
        
        prompt = f"""Score this IELTS speaking transcript. Return JSON:
{{
    "fluency": <score>,
    "lexical_resource": <score>,
    "grammatical_range": <score>,
    "pronunciation": <score>
}}

Transcript: {request.transcript}
"""
        
        try:
            resp = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=prompt
            )
            
            scores = json.loads(resp.text)
            
            return TranscriptScoreResponse(
                scores=scores,
                backend="gemini",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini API failed: {str(e)}"
            )


@router.get("/api/v1/model-info")
async def get_model_info():
    """Get information about available models"""
    server = get_inference_server()
    
    return {
        "local_model_available": server is not None,
        "local_model_path": os.getenv("LOCAL_MODEL_PATH", "./models/ielts-finetuned"),
        "inference_backend": os.getenv("INFERENCE_BACKEND", "vllm"),
        "gemini_available": True,
    }


# Add to main app
# from app.main import app
# app.include_router(router)
