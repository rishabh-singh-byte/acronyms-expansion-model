"""
Acronym extraction and multi-model inference service.
Extracts acronyms from queries, matches against dictionary, and dispatches to AI models.
"""

import json
import re
from typing import Dict, List
from app.models.vllm_client import call_vllm
from app.models.openai_client import call_openai

ACRONYM_FILE = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/acronyms_list_cleaned.json"

with open(ACRONYM_FILE, "r") as f:
    ACRONYMS = json.load(f)

def extract_acronyms(query: str) -> Dict[str, List[str]]:
    """
    Extract acronyms from query that exist in dictionary.
    
    Args:
        query: User input text
    
    Returns:
        Dict mapping found acronyms to their possible expansions
    """
    found = {}
    words = re.findall(r'\b[a-zA-Z]{1,}\b', query)
    for word in words:
        if word in ACRONYMS:
            found[word] = ACRONYMS[word]
    return found

def build_structured_prompt(query: str, found_acronyms: Dict[str, List[str]]) -> str:
    """
    Format query and acronyms into prompt for model.
    
    Args:
        query: Original user query
        found_acronyms: Dict of acronyms with their expansions
    
    Returns:
        Formatted prompt string
    """
    candidate_strs = [
        f"({acro}: {', '.join(expansions)})"
        for acro, expansions in found_acronyms.items()
    ]
    candidate_section = " ".join(candidate_strs)
    return f'query: "{query}", candidate acronyms: "{candidate_section}"'

async def get_all_model_responses(
    query: str,
    use_qwen_base: bool = True,
    use_qwen_lora: bool = True,
    use_openai_gpt: bool = True
) -> Dict:
    """
    Process query through selected AI models for acronym expansion.
    
    Args:
        query: User query text
        use_qwen_base: Enable Qwen base model
        use_qwen_lora: Enable Qwen LoRA model
        use_openai_gpt: Enable OpenAI GPT model
    
    Returns:
        Dict with query, found acronyms, and model results
    """
    found_acronyms = extract_acronyms(query)

    if not found_acronyms:
        return {
            "query": query,
            "acronyms_found": {},
            "results": {
                "qwen_base": "No known acronyms found." if use_qwen_base else None,
                "qwen_lora": "No known acronyms found." if use_qwen_lora else None,
                "openai_gpt": "No known acronyms found." if use_openai_gpt else None
            }
        }

    user_query = build_structured_prompt(query, found_acronyms)
    results = {}

    if use_qwen_base:
        base_resp = await call_vllm(user_query, use_lora=False)
        try:
            results["qwen_base"] = json.loads(base_resp)
        except Exception:
            results["qwen_base"] = base_resp

    if use_qwen_lora:
        lora_resp = await call_vllm(user_query, use_lora=True)
        try:
            results["qwen_lora"] = json.loads(lora_resp)
        except Exception:
            results["qwen_lora"] = lora_resp

    if use_openai_gpt:
        openai_resp = await call_openai(user_query)
        try:
            results["openai_gpt"] = json.loads(openai_resp)
        except Exception:
            results["openai_gpt"] = openai_resp

    return {
        "query": query,
        "acronyms_found": found_acronyms,
        "results": results
    }
