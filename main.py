"""
FastAPI application entry point for the Acronym Expansion API.
Provides endpoints for context-aware acronym expansion using multiple AI models.
"""

from fastapi import FastAPI
from app.routes.run_inference import router as inference_router

app = FastAPI(
    title="Acronym Explanation API",
    description="An API to extract acronyms and call multiple LLMs (vLLM base, LoRA, OpenAI) for expanded understanding.",
    version="1.0.0",
    port=8090
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Acronym Explanation API is up and running!"}

app.include_router(inference_router, prefix="/inference", tags=["Inference"])
