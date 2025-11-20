# app/routes/run_inference.py
"""
API routes for multi-model inference endpoints.
Handles random query generation and acronym expansion across multiple AI models.
"""

from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter
from app.services.input_query import get_all_model_responses_random

class QueryRequest(BaseModel):
    """Request model for inference endpoint"""
    n: int = 5
    use_qwen_base: Optional[bool] = True
    use_qwen_lora: Optional[bool] = True
    use_openai_gpt: Optional[bool] = True
    use_tiny_llama_lora: Optional[bool] = False

router = APIRouter()

@router.post("/generate")
async def generate(request: QueryRequest):
    """
    Generate acronym expansions for n random queries using selected models.
    
    Args:
        request: QueryRequest with model selection flags
    
    Returns:
        Dict with total_samples and data list containing results per query
    """
    return await get_all_model_responses_random(
        n=request.n,
        use_qwen_base=request.use_qwen_base,
        use_qwen_lora=request.use_qwen_lora,
        use_openai_gpt=request.use_openai_gpt,
        use_tiny_llama_lora=request.use_tiny_llama_lora
    )