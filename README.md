# Acronym Expansion API

AI-powered acronym expansion system using multiple fine-tuned language models for context-aware disambiguation.

## Overview

This FastAPI application automatically detects acronyms in user queries and selects the most contextually relevant expansion using multiple AI models:
- **Qwen 4B** (base & LoRA fine-tuned)
- **TinyLlama 1.1B** (LoRA fine-tuned)
- **OpenAI GPT-4o-mini** (baseline)

### Core Functionality
- **Intelligent Acronym Extraction**: Automatically identifies and extracts acronyms from user input
- **Multi-Model Inference**: Simultaneously processes queries through four different AI models for comparison
  - Includes base model (Qwen) and three fine-tuned models (Qwen QLora, Tiny Llama QLora, OpenAI GPT)
- **Contextual Understanding**: Selects the most relevant acronym expansion based on query context using fine-tuned adapters
- **Curated Dictionary**: Leverages a comprehensive acronym dictionary with multiple expansion options
- **Flexible Query Input**: Supports both specific user queries and random query generation for testing

### AI Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Qwen Base** | Qwen3-4B-Instruct-FP8 base model | General-purpose acronym expansion |
| **Qwen LoRA** | Qwen with fine-tuned QLora adapter | Domain-specific acronym understanding |
| **Tiny Llama LoRA** | TinyLlama-1.1B with fine-tuned QLora adapter | Efficient inference with domain adaptation |
| **OpenAI GPT** | Azure GPT-4o-mini | Baseline comparison and validation |

## Architecture

```
              ┌─────────────────┐
              │   User Query    │
              └────────┬────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Acronym Extraction     │
          │  (Pattern Matching)     │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Dictionary Lookup      │
          │  (Candidate Expansions) │
          └────────────┬────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│          Multi-Model Inference                  │
│  ┌──────────┬──────────┬──────────┬──────────┐  │
│  │  Qwen    │  Qwen    │  Tiny    │ OpenAI   │  │ 
│  │  Base    │  LoRA    │  Llama   │   GPT    │  │
│  │          │          │  LoRA    │          │  │
│  └──────────┴──────────┴──────────┴──────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
                  ┌─────────┐
                  │ Results │
                  └─────────┘
```

## Prerequisites

- **Python**: 3.8 or higher
- **vLLM Server**: Running instance with Qwen and Tiny Llama models
- **Azure OpenAI**: Valid API credentials
- **System Requirements**: Minimum 4GB RAM, recommended 8GB+

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd /path/to/acronyms_expansion_model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

# Configure environment
# Update config.py with your Azure OpenAI credentials
```

### Running

**Option 1: Full Stack (API + UI)**
```bash
# Terminal 1: Start API
uvicorn app.main:app --reload --port 8090

# Terminal 2: Start UI
streamlit run app/streamlit/app1.py
```

**Option 2: Standalone UI (No API)**
```bash
streamlit run app/streamlit/app3.py
```

Access UI at `http://localhost:8501`

## API Endpoints

### Health Check
```bash
GET /
```

### Generate Expansions
```bash
POST /inference/generate
```

**Request:**
```json
{
  "n": 5,
  "use_qwen_base": true,
  "use_qwen_lora": true,
  "use_openai_gpt": true,
  "use_tiny_llama_lora": false
}
```

**Response:**
```json
{
  "total_samples": 5,
  "data": [{
    "query": "What does AI team do?",
    "candidate_acronyms": "(AI: artificial intelligence, Action Items)",
    "results": {
      "qwen_base": {"AI": ["artificial intelligence"]},
      "qwen_lora": {"AI": ["artificial intelligence"]},
      "tiny_llama_lora": {"AI": ["artificial intelligence"]},
      "openai_gpt": {"AI": ["artificial intelligence"]}
    }
  }]
}
```

**Interactive Docs:**
- Swagger UI: `http://localhost:8090/docs`
- ReDoc: `http://localhost:8090/redoc`

## Project Structure

```
app/
├── main.py                    # FastAPI entry point
├── instruction.txt            # Setup and running instructions
├── models/                    # AI model clients
│   ├── vllm_client.py        # Qwen model client
│   ├── tinyllama_client.py   # TinyLlama client
│   ├── openai_client.py      # Azure OpenAI client
│   └── prompt.py             # System prompts
├── routes/                    # API endpoints
│   └── run_inference.py
├── services/                  # Business logic
│   ├── acronyms_service.py   # Acronym extraction
│   └── input_query.py        # Query sampling
├── streamlit/                 # Web interfaces
│   ├── app.py                # Single query UI
│   ├── app1.py               # Evaluation UI
│   └── app3.py               # Standalone UI
├── evaluation_v1/             # Model evaluation scripts
│   ├── call_llama.py                # calling llama(on 20k samples)
│   ├── gpt_llama_evaluation.py      # Evaluation on llama output using gpt(judge)
│   └── gpt_qwen_evaluation.py       # Evaluation on qwen output using gpt(judge)
│   └── qwen_base_inference.py       # calling qwen base/lora on 20k samples
|
```

## Model Information

### Qwen 4B
- **Base**: `Qwen3-4B-Instruct-2507-FP8`
- **LoRA**: Fine-tuned on 16K acronym samples
- **Config**: r=8, alpha=16, dropout=0.05

### TinyLlama 1.1B
- **Base**: `TinyLlama-1.1B-Chat-v1.0`
- **LoRA**: Fine-tuned on 16K acronym samples
- **Config**: r=8, alpha=16, dropout=0.05

### OpenAI GPT
- **Model**: `gpt-4o-mini` via Azure
- **Purpose**: Baseline comparison

## Configuration

### Azure OpenAI (config.py)
```python
AZURE_API_KEY = "your-api-key"
AZURE_ENDPOINT = "your-endpoint"
AZURE_API_VERSION = "2024-08-01-preview"
```

### vLLM Server
Default URL: `http://98.89.19.168:8000`

Update in:
- `app/models/vllm_client.py`
- `app/models/tinyllama_client.py`

## Usage Examples

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8090/inference/generate",
    json={
        "n": 3,
        "use_qwen_base": True,
        "use_qwen_lora": True,
        "use_openai_gpt": True
    }
)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8090/inference/generate" \
  -H "Content-Type: application/json" \
  -d '{"n": 1, "use_qwen_base": true, "use_qwen_lora": true}'
```

## Data Files

See [DATA.md](DATA.md) for detailed information about:
- Acronym dictionary structure
- Dataset format
- Training data specifications

## Troubleshooting

<<<<<<< HEAD
### Common Issues
1. **vLLM Server Not Running**: Ensure vLLM server is running on `localhost:8000`
2. **Azure OpenAI Errors**: Check API credentials.
3. **File Path Errors**: Update absolute paths in configuration files
4. **Port Conflicts**: Change port in uvicorn command if 8090 is occupied
=======
### Port Already in Use
```bash
# Use different port
uvicorn app.main:app --reload --port 8091
>>>>>>> 0f604d5 (refactored structure)

# Kill existing process
lsof -ti:8090 | xargs kill -9
```

### vLLM Connection Failed
```bash
# Test connection
curl http://98.89.19.168:8000

# Update URL in model clients if needed
```

### Module Import Errors
```bash
# Run from project root
cd /path/to/ai-search-retrieval-pipeline-poc-2
PYTHONPATH=. uvicorn app.main:app --reload --port 8090
```

## Development

### Adding New Model
1. Create client in `app/models/`
2. Update `app/services/input_query.py`
3. Add parameter to `app/routes/run_inference.py`

### Running Evaluation
```bash
cd app/evaluation_v1
python qwen_base_inference.py
python gpt_qwen_evaluation.py
```

### Code Quality
```bash
black app/
flake8 app/
mypy app/
```

## vLLM Server Setup

See `instruction.txt` for detailed vLLM hosting commands.

**Quick Reference:**
```bash
# Qwen 4B
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --enable-lora \
  --lora-modules acronym-lora=/path/to/checkpoints/final_2 \
  --port 8000

# TinyLlama 1.1B
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --enable-lora \
  --lora-modules acronym-lora=/path/to/checkpoints/llama-1.1b-lora_2 \
  --port 8000
```

---

**Quick Links:**
- [Data Documentation](DATA.md)
- [API Docs](http://localhost:8090/docs) (when running)
- [Setup Instructions](instruction.txt)
