# app/models/openai_client.py
"""
Azure OpenAI client for GPT-4o-mini model inference.
Used as baseline comparison for acronym expansion accuracy.
"""

from openai import AsyncAzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION
from app.models.prompt import SYSTEM_PROMPT

OPENAI_MODEL = "gpt-4o-mini"

async def call_openai(user_query: str) -> str:
    """
    Call Azure OpenAI GPT model.
    
    Args:
        user_query: Formatted query with candidate acronyms
        
    Returns:
        Model response as JSON string or error message
    """
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