# app/services/input_query.py
"""
Random query sampling service for model evaluation.
Samples queries from dataset and dispatches to multiple AI models for comparison.
"""

import json
import random
from typing import Dict, Any, List
from app.models.vllm_client import call_vllm
from app.models.openai_client import call_openai
from app.models.tinyllama_client import call_tinyllama

DATA_FILE = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/golden_data_20k.json"

with open(DATA_FILE, "r") as f:
    DATASET = json.load(f)

def sample_queries(n: int) -> List[Dict[str, Any]]:
    """
    Sample random queries from dataset.
    
    Args:
        n: Number of queries to sample
    
    Returns:
        List of n random query entries
    """
    return random.sample(DATASET, min(n, len(DATASET)))

async def get_all_model_responses_random(
    n: int = 5,
    use_qwen_base: bool = True,
    use_qwen_lora: bool = True,
    use_openai_gpt: bool = True,
    use_tiny_llama_lora: bool = False
) -> Dict[str, Any]:
    """
    Sample n queries and process through selected AI models.
    
    Args:
        n: Number of queries to sample
        use_qwen_base: Enable Qwen base model
        use_qwen_lora: Enable Qwen LoRA model
        use_openai_gpt: Enable OpenAI GPT model
        use_tiny_llama_lora: Enable TinyLlama LoRA model
    
    Returns:
        Dict with total_samples count and data list of results
    """
    samples = sample_queries(n)
    all_results = []

    for item in samples:
        query = item.get("Query", "")
        candidate_acronyms = item.get("Candidate_Acronyms", "")
        formatted_query = f'query: "{query}", candidate acronyms: "{candidate_acronyms}"'

        result_entry = {
            "query": query,
            "candidate_acronyms": candidate_acronyms,
            "results": {}
        }

        # --- Qwen base ---
        if use_qwen_base:
            base_resp = await call_vllm(formatted_query, use_lora=False)
            try:
                result_entry["results"]["qwen_base"] = json.loads(base_resp)
            except Exception:
                result_entry["results"]["qwen_base"] = base_resp

        # --- Qwen LoRA ---
        if use_qwen_lora:
            lora_resp = await call_vllm(formatted_query, use_lora=True)
            try:
                result_entry["results"]["qwen_lora"] = json.loads(lora_resp)
            except Exception:
                result_entry["results"]["qwen_lora"] = lora_resp

        # --- OpenAI GPT ---
        if use_openai_gpt:
            openai_resp = await call_openai(formatted_query)
            try:
                result_entry["results"]["openai_gpt"] = json.loads(openai_resp)
            except Exception:
                result_entry["results"]["openai_gpt"] = openai_resp

        # --- TinyLlama LoRA ---
        if use_tiny_llama_lora:
            tinyllama_resp = await call_tinyllama(formatted_query, use_lora=True)
            try:
                result_entry["results"]["tinyllama_lora"] = json.loads(tinyllama_resp)
            except Exception:
                result_entry["results"]["tinyllama_lora"] = tinyllama_resp

        all_results.append(result_entry)

    return {"total_samples": len(all_results), "data": all_results}
