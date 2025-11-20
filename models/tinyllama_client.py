# app/models/tinyllama_client.py
"""
TinyLlama client for lightweight model inference via vLLM.
Supports both base model and LoRA adapter for resource-efficient acronym expansion.
"""

import httpx
from app.models.prompt import SYSTEM_PROMPT

TINYLLAMA_API_URL = "http://98.89.19.168:8000/v1/chat/completions"
TINYLLAMA_BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_LORA_ADAPTER_NAME = "acronym-lora"

async def call_tinyllama(user_query: str, use_lora: bool = False) -> str:
    """
    Call TinyLlama model via vLLM API.
    
    Args:
        user_query: Formatted query with candidate acronyms
        use_lora: If True, uses fine-tuned LoRA adapter; otherwise base model
    
    Returns:
        Model response as JSON string or error message
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    model_name = TINYLLAMA_LORA_ADAPTER_NAME if use_lora else TINYLLAMA_BASE_MODEL_NAME
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 400
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(TINYLLAMA_API_URL, json=payload)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error - TinyLlama {'LoRA' if use_lora else 'Base'}]: {e}"

