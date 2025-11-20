# app/streamlit/app3.py
"""
Standalone Streamlit interface for acronym expansion.
Runs complete inference pipeline without requiring FastAPI server.
Directly calls model APIs for evaluation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
import random
import asyncio
import httpx
from openai import AsyncAzureOpenAI

DATA_FILE = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/golden_data_20k.json"

VLLM_API_URL = "http://98.89.19.168:8000/v1/chat/completions"
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
LORA_ADAPTER_NAME = "acronym-lora"

TINYLLAMA_API_URL = "http://98.89.19.168:8000/v1/chat/completions"
TINYLLAMA_BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_LORA_ADAPTER_NAME = "acronym-lora"

AZURE_ENDPOINT = "https://ai-sa-api-dev.openai.azure.com/"
AZURE_API_KEY = "a9d5a4c2ce944d2b856481406847e286"
AZURE_API_VERSION = "2024-08-01-preview"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a precise assistant tasked with selecting only the **most relevant acronym expansions** from a given list, based strictly on the user's query.

Instructions:
- Only include expansions that are clearly and directly related to the query's context.
- If multiple meanings are relevant, include all of them.
- If no acronym is relevant, return an empty dictionary: `{}`.
- Acronyms must appear in the query to be considered.
- Preserve the acronym casing as it appears in the query.
- Output must be a valid **JSON dictionary**:
  - Keys: acronyms found in the query.
  - Values: lists of relevant expansions (as strings).

Output Format:
{
  "ACRONYM1": ["Relevant Expansion 1", "Relevant Expansion 2",...],
  "ACRONYM2": ["Relevant Expansion 1", "Relevant Expansion 2",...],
}

Examples:
###
query: "Who leads the AI team", candidate acronyms: " (AI: artificial intelligence, Artificial Intelligence, Action Items)"
###
{"AI": ["artificial intelligence"]}
###
query: "who is the current cpo", candidate acronyms: " (cpo: Chief People Officer, Chief Product and Customer Officer, Chief Product Officer)"
###
{"cpo": ["Chief People Officer", "Chief Product Officer"]}
###
query: "update the okr", candidate acronyms: " (okr: Objectives and Key Results, Office of Knowledge Research)"
###
{"okr": ["Objectives and Key Results"]}
###
query: "can you help me with this", candidate acronyms: " (can: Canada) (you: Young Outstanding Undergraduates)"
###
{}
###
"""

@st.cache_data
def load_dataset():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

DATASET = load_dataset()

def sample_queries(n: int):
    return random.sample(DATASET, min(n, len(DATASET)))

async def call_vllm(user_query: str, use_lora: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    model_name = LORA_ADAPTER_NAME if use_lora else BASE_MODEL_NAME
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 400
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(VLLM_API_URL, json=payload)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error - vLLM {'LoRA' if use_lora else 'Base'}]: {e}"

async def call_openai(user_query: str) -> str:
    try:
        client = AsyncAzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )

        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error - OpenAI]: {e}"

async def call_tinyllama(user_query: str, use_lora: bool = False) -> str:
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

async def run_inference(
    n: int,
    use_qwen_base: bool,
    use_qwen_lora: bool,
    use_openai_gpt: bool,
    use_tiny_llama_lora: bool = False
):
    samples = sample_queries(n)
    results = []

    for item in samples:
        query = item.get("Query", "")
        candidate_acronyms = item.get("Candidate_Acronyms", "")
        formatted_query = f'query: "{query}", candidate acronyms: "{candidate_acronyms}"'

        entry = {
            "query": query,
            "candidate_acronyms": candidate_acronyms,
            "results": {}
        }

        tasks = []
        if use_qwen_base:
            tasks.append(call_vllm(formatted_query, use_lora=False))
        if use_qwen_lora:
            tasks.append(call_vllm(formatted_query, use_lora=True))
        if use_openai_gpt:
            tasks.append(call_openai(formatted_query))
        if use_tiny_llama_lora:
            tasks.append(call_tinyllama(formatted_query, use_lora=True))

        model_names = []
        if use_qwen_base: model_names.append("qwen_base")
        if use_qwen_lora: model_names.append("qwen_lora")
        if use_openai_gpt: model_names.append("openai_gpt")
        if use_tiny_llama_lora: model_names.append("tinyllama_lora")

        responses = await asyncio.gather(*tasks)

        for model_name, resp in zip(model_names, responses):
            try:
                parsed = json.loads(resp)
            except Exception:
                parsed = resp
            entry["results"][model_name] = parsed

        results.append(entry)

    return results

st.set_page_config(page_title="Acronym Expansion UI", layout="wide")

st.title("🤖 Acronym Expansion Assistant")

st.sidebar.header("⚙️ Model Configuration")

n_samples = st.sidebar.number_input(
    "Number of random queries", 
    min_value=1, 
    max_value=50, 
    value=3,
    step=1
)

use_qwen_base = st.sidebar.checkbox("Use Qwen Base", True)
use_qwen_lora = st.sidebar.checkbox("Use Qwen LoRA Adapter", True)
use_openai_gpt = st.sidebar.checkbox("Use OpenAI GPT", True)
use_tiny_llama_lora = st.sidebar.checkbox("Use TinyLlama LoRA Adapter", False)

if st.button("🚀 Run Inference", use_container_width=True):
    with st.spinner("Running model inference... this might take a while ⏳"):
        results = asyncio.run(
            run_inference(
                n=n_samples,
                use_qwen_base=use_qwen_base,
                use_qwen_lora=use_qwen_lora,
                use_openai_gpt=use_openai_gpt,
                use_tiny_llama_lora=use_tiny_llama_lora
            )
        )

    st.success("✅ Inference Complete!")

    for i, res in enumerate(results, 1):
        with st.expander(f"📄 Query {i}: {res['query']}"):
            st.markdown(f"**Query:** {res['query']}")
            st.markdown(f"**Candidate Acronyms:** {res['candidate_acronyms']}")

            for model_name, output in res["results"].items():
                with st.expander(f"🧠 {model_name.replace('_', ' ').title()}"):
                    if isinstance(output, dict):
                        for k, v in output.items():
                            st.markdown(f"- **{k}** → {', '.join(v) if isinstance(v, list) else v}")
                    else:
                        st.code(output, language="json")

st.markdown("---")
st.caption("Built using Streamlit")
