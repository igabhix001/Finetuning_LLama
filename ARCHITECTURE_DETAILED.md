# AI Astrologer — Detailed System Architecture (Training → Serving → RAG)

## 1) Repository at a Glance

### 1.1 Top-level structure (key paths)

- **`configs/`**
  - `dapt_config.yaml`, `sft_config.yaml`, `dpo_config.yaml` (+ LoRA configs)
- **`data/`**
  - `dapt_corpus/` (HF dataset, DAPT)
  - `sft_train/`, `sft_validation/` (HF datasets, SFT)
  - `dpo/dpo_pairs.jsonl` (raw DPO pairs)
  - `dpo/prepared/` (HF dataset for TRL DPOTrainer)
  - `pinecone_upsert.jsonl` (KB vectors to upsert)
- **`scripts/`**
  - Training: `03_train_dapt.py`, `04_train_sft.py`, `05b_merge_sft_lora.py`, `15_train_dpo.py`, `16_merge_dpo_lora.py`
  - Quantize/export: `06_quantize_unsloth.py`
  - Serving: `08_serve_vllm.py` (vLLM OpenAI-compatible server)
  - Apps: `09_chat_ui.py` (Gradio), `11_api_server.py` (FastAPI)
  - RAG indexing: `02_upload_pinecone.py` (KB), `11_enrich_kb.py` (KB metadata), `12_build_product_index.py` (products)
  - DPO dataset generation/prep: `13_generate_dpo_dataset.py`, `14_prepare_dpo_dataset.py`
  - Evaluation: `10_kp_test_suite.py` (standard + production-mirror mode)
  - Dataset maintenance: `17_renormalize_sft_dataset.py` (SFT output cleanup), `18_clean_dapt_corpus.py` (DAPT boilerplate/OCR removal)

### 1.2 Runtime services (process-level)

- **Service A: vLLM inference server**
  - Script: `scripts/08_serve_vllm.py`
  - Exposes: OpenAI-compatible endpoints at `http://<host>:<port>/v1/...`
- **Service B: FastAPI “product + RAG + postprocess” API**
  - Script: `scripts/11_api_server.py`
  - Exposes: `POST /chat`, `GET /health`
  - Depends on: vLLM, optional Pinecone + OpenAI embeddings
- **Service C: Gradio Chat UI**
  - Script: `scripts/09_chat_ui.py`
  - Depends on: vLLM, optional Pinecone + OpenAI embeddings


## 2) End-to-End Data & Control Flow

### 2.1 High-level flow

```
User (JSON kundali) + Question
  -> Gradio UI (09_chat_ui.py) OR FastAPI (11_api_server.py)
      -> chart_preprocessor.chart_to_yaml(JSON)  [shared]
      -> (optional) Pinecone RAG retrieval (kp-astrology-kb)
      -> (optional) Pinecone product retrieval (kp-products) for remedies only
      -> Build OpenAI-style chat request
      -> vLLM OpenAI-compatible API (/v1/chat/completions)
      -> Postprocess (strip markdown/headers, date sanity, health safety, etc.)
      -> Enrich (Hindi quote + product sentence if needed)
      -> Return final response
```

### 2.2 “Training ↔ Inference format consistency” invariant

**Single source of truth:** `scripts/chart_preprocessor.py`.

- Both training-time DPO generation (`13_generate_dpo_dataset.py`) and runtime apps (`09_chat_ui.py`, `11_api_server.py`) call the same `chart_to_yaml()`.
- This is a critical architecture decision because the project’s core failure mode was **training/inference mismatch** (model trained on one chart representation but asked to infer on another).


## 3) Chart Preprocessing Subsystem (`scripts/chart_preprocessor.py`)

### 3.1 Purpose

Transform a very large computation-engine kundali JSON into a compact YAML representation that:

- Fits inside the model context budget reliably
- Preserves KP-critical fields (sign/star/sub + house significations)
- Adds time-awareness (`today_date`, `age_now`)
- Keeps dasha hierarchy with **pratyantar dashas** for timing precision

### 3.2 Output schema (conceptual)

- `today_date: DD Mon YYYY`
- `native:`
  - `name`, `gender`, `dob`, `tob`, `place`, `lagna`, `rasi`, `nakshatra`, `age_now`
- `planets:` (KP planets only)
  - `Planet: sign | nakshatra (nak_lord) | sub: sub_lord | houses: [...]`
- `cusps:`
  - `1..12: sign | nakshatra (nak_lord) | sub: sub_lord`
- `significators:`
  - `1..12: [planets...]`
- `dashas:`
  - balance
  - previous MD (last 4 ADs)
  - current MD (all ADs + selected PDs)
  - next MD (first AD + PDs)
- Derived convenience groups:
  - `Significators_for_Marriage (houses 2,7,11): ...`
  - `Significators_for_Career (houses 2,6,10): ...`
  - `Significators_for_Finance (houses 2,6,11): ...`

### 3.3 Budget control

- Hard cap: `MAX_CHART_YAML = 14000` characters
- If chart YAML exceeds cap: truncated with `# ...truncated`


## 4) Retrieval-Augmented Generation (RAG)

### 4.1 Knowledge base (KP books) index

- Pinecone index: **`kp-astrology-kb`** (default)
- Script to upload: `scripts/02_upload_pinecone.py`
- Embedding model: **OpenAI `text-embedding-3-large`**, dim **3072**

**Query flow (runtime):**

1. Embed user question (truncated to ~500 chars)
2. Pinecone `query(top_k=K, include_metadata=True)`
3. Format retrieved chunks as:
   - `[{rule_refs}](Source: book, page) chunk_text`
4. Inject into the system prompt under a “KP Book Excerpts” section

### 4.2 KB enrichment

- Script: `scripts/11_enrich_kb.py`
- Purpose: enrich chunk metadata (e.g., page numbers) + ensure missing topics are chunked


## 5) Product Recommendation Index (Remedy-only)

### 5.1 Index

- Pinecone index: **`kp-products`** (default)
- Script: `scripts/12_build_product_index.py`
- Input: client product export CSV (e.g. `products_export_2026-02-03.csv`)

### 5.2 Runtime logic (API + UI)

- Products are used **only when the question is classified as a remedy query** (keyword-based).
- Two retrieval strategies:
  - **Primary:** Pinecone semantic search on `kp-products` using OpenAI embeddings
  - **Fallback:** CSV keyword matching if Pinecone product index is not available


## 6) Serving & Application Layer

### 6.1 vLLM Serving (`scripts/08_serve_vllm.py`)

- Launches `python -m vllm.entrypoints.openai.api_server`
- Flags:
  - `--model <path>`
  - `--served-model-name kp-astrology-llama`
  - `--max-model-len` (default from env/CLI)
  - `--dtype bfloat16`

### 6.2 Gradio UI (`scripts/09_chat_ui.py`)

- Accepts pasted kundali JSON in a left-side textbox
- Converts JSON → YAML via `_chart_to_yaml()`
- Retrieves RAG chunks + (remedy-only) products
- Streams vLLM tokens, continuously postprocessing partial output

### 6.3 FastAPI Server (`scripts/11_api_server.py`)

- Endpoint: `POST /chat`
  - Input: `{ question: str, chart_data?: str }`
  - Output: `{ answer: str, prediction?: str, product_reco?: {sku,title,price} }`
- Implements:
  - Query classification
  - Character-budget-based RAG trimming
  - Postprocessing (format + safety)
  - Optional enrichment (Hindi quote; product sentence if remedy query)

### 6.4 Context window budgeting (operational invariant)

Both UI and API follow the same logic:

- Choose `OUTPUT_TOKENS`
- Estimate input tokens from char count (`~0.78 chars/token` heuristic)
- Compute `available = MAX_MODEL_LEN - est_input_tokens`
- Set `max_tokens = min(base_output, available)`

This is the system’s primary defense against vLLM context-length errors.


## 7) Postprocessing & Safety (Runtime)

### 7.1 Postprocess goals

- Strip markdown (`**bold**`, headings, code blocks)
- Strip “robotic” section headers (“Analysis:”, “Conclusion:”, etc.)
- Normalize dates (ISO → “Mon YYYY”)
- Enforce date sanity relative to user birth year
- Replace dangerous medical terms with safer phrasing
- Cap paragraphs and sentences

### 7.2 Enrichment

- Append a short Hindi motivational quote if missing
- Append *one* product suggestion sentence only if remedy query AND model didn’t include one


## 8) Training System

### 8.1 Stage 1: DAPT LoRA (`scripts/03_train_dapt.py` + `configs/dapt_config.yaml`)

- Input dataset: `data/dapt_corpus/` (HF dataset on disk)
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Output: LoRA adapters in `checkpoints/dapt_lora/`

### 8.2 Stage 2: SFT LoRA (`scripts/04_train_sft.py` + `configs/sft_config.yaml`)

- Input datasets:
  - `data/sft_train/`
  - `data/sft_validation/`
- Output: LoRA adapters in `checkpoints/sft_lora/`

### 8.3 Stage 2→3 bridge: merge DAPT+SFT into a single base (`scripts/05b_merge_sft_lora.py`)

- Loads base model + merges DAPT LoRA + merges SFT LoRA
- Produces: `models/merged_sft/`

### 8.4 Stage 3: DPO LoRA (`scripts/15_train_dpo.py` + `configs/dpo_config.yaml`)

- Input dataset: `data/dpo/prepared/` (HF dataset)
- Trainer: TRL `DPOTrainer`
- Output: `checkpoints/dpo_lora/final/`

### 8.5 Final merge: DPO LoRA into merged base (`scripts/16_merge_dpo_lora.py`)

- Produces final deployment model: `models/final_dpo/`

### 8.6 DPO dataset generation (OpenAI Batch API)

- `scripts/13_generate_dpo_dataset.py`
  - Auto-discovers `kundali_*.json`
  - Converts each chart via `chart_to_yaml()`
  - Generates chosen/rejected pairs, optionally with GPT-as-judge scoring
  - Writes: `data/dpo/dpo_pairs.jsonl`
- `scripts/14_prepare_dpo_dataset.py`
  - Converts raw JSONL pairs to HF dataset for DPOTrainer
  - Writes: `data/dpo/prepared/`


## 9) Environment Variables / Configuration Surface

### 9.1 `.env` / `.env.example`

- `PINECONE_API_KEY`
- `PINECONE_ENVIRONMENT`
- `PINECONE_INDEX_NAME` (default: `kp-astrology-kb`)
- `OPENAI_API_KEY`
- `HF_TOKEN`

### 9.2 Runtime CLI surface

- UI: `09_chat_ui.py --vllm-url --port --no-rag --top-k --max-model-len --products-csv`
- API: `11_api_server.py --vllm-url --port --no-rag --top-k --max-model-len --products-csv`
- vLLM: `08_serve_vllm.py --model-path --host --port --max-model-len --gpu-memory-utilization`


## 10) Dataset Quality Snapshot (as observed in-repo)

### 10.1 DAPT (`data/dapt_corpus/`)

- Count: **654**
- Typical text length: **~7.2k chars** (median ~7.3k)
- Notable: a small number of “Internet Archive / EPUB / OCR notice” artifacts exist

### 10.2 SFT (`data/sft_train/`, `data/sft_validation/`)

- Train count: **19,303**
- Language split: **15,000 en / 4,303 hi**
- Strong signal: outputs frequently contain markdown/bold, headers, and “the native” phrasing

### 10.3 DPO (`data/dpo/dpo_pairs.jsonl` → `data/dpo/prepared/`)

- Raw pairs: **999**
- Prepared train/eval: **895 / 100**
- Chosen outputs: ~3 sentences average, 95th percentile is 4 sentences
- Chosen outputs are format-clean (no bold/headers detected in scan)



## 11) Recommended “Architecture Contracts” to Treat as Non-Negotiable

- **Contract A: chart format**
  - All training and inference must consume the same YAML schema from `chart_preprocessor.py`.
- **Contract B: response format**
  - No markdown, no headers, short answers; enforce with both postprocessing and DPO.
- **Contract C: product discipline**
  - Only in remedy queries (and exactly one product, maximum).
- **Contract D: context budgeting**
  - Any change to prompts/RAG must preserve `MAX_MODEL_LEN` safety margins.
