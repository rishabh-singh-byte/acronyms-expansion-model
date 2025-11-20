# Data Documentation

This document inventories every JSON file that lives under `app/` (both the root `data/` folder and `evaluation_v1/results/`).  
For each asset we describe the schema, usage, and where it originates so the repository is audit ready.

---

## JSON Inventory (quick reference)

| # | File | Records* | Data shape | Snapshot purpose / origin |
|---|------|----------|------------|---------------------------|
| 1 | `app/data/acronyms_list_cleaned.json` | 8.5k acronyms (59k expansions) | `dict[str, list[str]]` | Lookup dictionary exported via `Notebooks/acronyms_extraction.ipynb` |
| 2 | `app/data/golden_data_20k.json` | 20k rows | `list[{"Query","Candidate_Acronyms","Best_Output"}]` | Canonical golden set sampled for demos, QA, and regression |
| 3 | `app/data/sampled_20000_queries.json` | 20k rows | `list[{"query","candidate_acronyms","output"}]` | Normalized sampler output used by bulk inference jobs |
| 4 | `app/evaluation_v1/results/mismatched_outputs_qwen_base.json` | 10,921 rows | `list[{"index","Query","model_1","model_2"}]` | Raw Qwen-base mismatches exported from `app1/test.ipynb` |
| 5 | `app/evaluation_v1/results/mismatched_outputs_qwen(ft).json` | 4,488 rows | same schema | Raw Qwen LoRA mismatches from `app1/test.ipynb` |
| 6 | `app/evaluation_v1/results/mismatched_outputs_llama_2nd.json` | 5,621 rows | `list[{"index","Query","model_1_gpt","model_2_llama"}]` | TinyLlama vs GPT mismatches (Excel diff in `app1/test.ipynb`) |
| 7 | `app/evaluation_v1/results/mismatched_evaluation_results_gpt_base.json` | 10,921 rows | `list[{"query","model_1_qwen_base","model_2_gpt",...}]` | GPT-4o-mini verdicts (Model 1=Qwen) from `gpt_qwen_evaluation.py` |
| 8 | `app/evaluation_v1/results/mismatched_evaluation_results_gpt_base_2.json` | 10,921 rows | same schema | Role-swapped rerun of #7 to remove ordering bias |
| 9 | `app/evaluation_v1/results/mismatched_evaluation_results_gpt_qwen(ft)1.json` | 4,488 rows | same schema | GPT vs Qwen-QLoRA judgments (Model 1=GPT) |
| 10 | `app/evaluation_v1/results/mismatched_evaluation_results_gpt_qwen(ft)2.json` | 4,488 rows | same schema | Role-swapped rerun of #9 |

---

## Foundational datasets (`app/data/`)

### `app/data/acronyms_list_cleaned.json`
- **Shape:** dictionary with 8,535 keys; each value is a de-duplicated list of candidate expansions (≈59K total items).
- **Schema:** `{ "<acronym>": ["expansion 1", "expansion 2", ...] }`. Entries retain the casing found in the source corpus so downstream prompts match user text.
- **Usage:** Loaded once on startup by the acronym service and used to extract candidates before any model call.

```12:32:app/services/acronyms_service.py
ACRONYM_FILE = "/Users/rishabh.singh/Desktop/ai-search-retrieval-pipeline-poc-2/app/acronyms_list_cleaned.json"
with open(ACRONYM_FILE, "r") as f:
    ACRONYMS = json.load(f)

def extract_acronyms(query: str) -> Dict[str, List[str]]:
    ...
```

- **Origin:** `Notebooks/acronyms_extraction.ipynb` scrapes, cleans, and exports this JSON (see notebook cell log “📄 Output written to : acronyms_list_cleaned.json”). Treat it as the canonical dictionary—any edits should flow through the notebook to keep provenance.

### `app/data/golden_data_20k.json`
- **Shape:** list of 20,000 dicts holding `Query`, a serialized `Candidate_Acronyms` string, and the curator-approved `Best_Output`.
- **Usage:** Sampled inside `services/input_query.py` and `streamlit/app3.py` to demo or regression-test model responses. The same file also seeds evaluation notebooks when we need reproducible prompts.
- **Schema example:**
  ```json
  {
    "Query": "ai quantexa",
    "Candidate_Acronyms": "(ai: artificial intelligence, analytical innovation, quantexa: quantexa solutions and platform products)",
    "Best_Output": "{\"AI\": [\"artificial intelligence\"], \"QUANTEXA\": [\"quantexa solutions and platform products\"]}"
  }
  ```
- **Origin:** This is the manually verified “golden set” exported from the internal labeling workflow before being checked into git, ensuring every evaluation run can be reproduced without hitting external storage.

### `app/data/sampled_20000_queries.json`
- **Shape:** list of 20,000 dicts enriched with normalized dictionaries (`candidate_acronyms` and `output` objects) so caller code never needs to parse strings.
- **Usage:** Direct input for bulk inference runners such as `evaluation_v1/qwen_base_inference.py` and `evaluation_v1/call_llama.py`; also used by analytics notebooks under `Notebooks/`.
- **Schema example:**
  ```json
  {
    "query": "worker comp",
    "candidate_acronyms": { "COMP": ["workers compensation"] },
    "output": { "COMP": ["workers compensation"] }
  }
  ```
- **Origin:** Generated in `Notebooks/acronyms_expansion.ipynb` (see log line “✅ Sampled 20000 queries and saved to: sampled_20000_queries.json”) by sampling from the 125k-query extraction and normalizing casing before shipping to `app/data/`.

---

## Evaluation pipeline overview

1. **Model sweep (XLSX stage):** Scripts like `evaluation_v1/qwen_base_inference.py`, `call_llama.py`, or Azure OpenAI runners produce Excel files that contain `expected_output` vs model responses for the 20K sampled queries.
2. **Mismatch extraction:** `app1/test.ipynb` loads those spreadsheets, lowercases + normalizes nested lists, and writes every disagreement to `mismatched_outputs_*.json`.
3. **Optional restructuring:** Some model outputs (TinyLlama) emit strings that combine multiple acronyms in a single key; `evaluation_v1/results/evaluation_gpt.ipynb` converts them into clean dicts (`converted_output_llama.json`) to ensure the judge sees comparable JSON.
4. **GPT judging:** `app/evaluation_v1/gpt_qwen_evaluation.py` (for Qwen vs GPT) and `app/evaluation_v1/gpt_llama_evaluation.py` (for TinyLlama vs GPT) call GPT-4o-mini / GPT-4-1-mini as neutral evaluators. Every mismatch file is evaluated twice—once per ordering—to neutralize positional bias (“position interchange” referenced in the project brief). The outputs are the `mismatched_evaluation_results_*.json` files listed below.

When you regenerate any stage, keep the filenames aligned; only the filenames change between base and LoRA runs—the scripts are identical.

---

## Raw mismatch files (`evaluation_v1/results/mismatched_outputs*.json`)

### `mismatched_outputs_qwen_base.json`
- **Shape:** 10,921 entries. Keys: `index`, `Query`, `model_1`, `model_2`.
- **Semantics:** `model_1` holds the lowercased expected dictionary; `model_2` contains Qwen-base predictions for cases where the dicts differ after normalization.
- **Usage:** Input to `gpt_qwen_evaluation.py` (Model 1=Qwen, Model 2=GPT) and to analytics notebooks (`analysis/explore_1.ipynb`) for failure taxonomy.
- **Origin:** Written by `app1/test.ipynb` Cell “CHECKING EXACT MISMATCHES COUNTS BETWEEN TWO MODELS (GPT AND QWEN-base)” by comparing `base_results_20000.xlsx`.

### `mismatched_outputs_qwen(ft).json`
- **Shape:** 4,488 entries (smaller mismatch set thanks to LoRA fine-tuning).
- **Semantics:** Same schema as above; this time `model_2` corresponds to the fine-tuned Qwen LoRA output captured in `best_model_outputs_1a.xlsx`.
- **Usage:** Fed twice to `gpt_qwen_evaluation.py` to create the two GPT-judged files below and to notebooks for diagnosis.
- **Origin:** `app1/test.ipynb` Cell “Total mismatches (QWEN-qlora)” writes the JSON to disk (filename renamed here to keep all artifacts under `app/evaluation_v1/results/`).

### `mismatched_outputs_llama_2nd.json`
- **Shape:** 5,621 entries; keys `model_1_gpt` and `model_2_llama` capture GPT ground truth vs TinyLlama LoRA response.
- **Usage:** Serves as the input for `gpt_llama_evaluation.py` (Azure GPT judge) and for subsequent restructuring via `evaluation_gpt.ipynb`.
- **Origin:** `app1/test.ipynb` Cell “Total mismatches (TinyLLAMA)” exports this JSON after diffing `llama1B_results_20_lora.xlsx`.

---

## GPT-judged mismatch verdicts (`evaluation_v1/results/mismatched_evaluation_results*.json`)

All of the following files share the same schema: each row stores the original `query`, the two model dicts (`model_1_*`, `model_2_*`), the merged `candidate_acronyms`, and GPT’s `better_model` plus a natural-language `justification`. Every dataset is evaluated twice with the model order flipped to implement the “position interchange” requirement noted in the project brief.

### Qwen base vs GPT
- **`mismatched_evaluation_results_gpt_base.json`** — Model 1 is Qwen-base and Model 2 is GPT. This pass answers “Does GPT think Qwen-base beat GPT when Qwen is shown first?”.
- **`mismatched_evaluation_results_gpt_base_2.json`** — Same content but with roles swapped (Model 1=GPT). Use this to confirm that verdicts remain stable regardless of ordering.
- **Generation:** Both files are emitted by `gpt_qwen_evaluation.py` (see `input_path` and `output_path` inside that script). Update the `output_path` when switching between `_base` and `_base_2`.

### Qwen LoRA (fine-tuned) vs GPT
- **`mismatched_evaluation_results_gpt_qwen(ft)1.json`** — Model 1=GPT, Model 2=Qwen-QLoRA; highlights where GPT still wins.
- **`mismatched_evaluation_results_gpt_qwen(ft)2.json`** — Roles swapped so Qwen-QLoRA appears as Model 1. Compare both to quantify gains independent of positional bias.
- **Generation:** Same `gpt_qwen_evaluation.py` script, simply pointing `input_path` at `mismatched_outputs_qwen(ft).json` and toggling the role-mapping helper.

### TinyLlama vs GPT (context)
- Although the judged TinyLlama outputs (`mismatched_evaluation_results_gpt_llama_nano*.json`) are stored at the repo root, they come from the same judge script (`gpt_llama_evaluation.py`).

---

## How to extend or regenerate these files

1. **Update source data** (e.g., rerun the extraction notebooks or regenerate XLSX evaluation dumps).
2. **Rebuild mismatched outputs** via `app1/test.ipynb` to capture fresh disagreements.
3. **Judge with GPT** by running `python app/evaluation_v1/gpt_qwen_evaluation.py` or `python app/evaluation_v1/gpt_llama_evaluation.py`. 
