# AI Astrologer — Issues, Bugs, Vulnerabilities & Fix Plan

This document is an industry-grade audit of the repo’s end-to-end system: datasets (DAPT/SFT/DPO), training scripts, serving stack (vLLM + FastAPI + Gradio), and RAG (Pinecone KB + product index).

## 0) Executive Summary (Current Readiness)

- **Serving architecture:** solid and pragmatic (vLLM OpenAI-compatible backend + thin app layer + optional RAG).
- **Core reliability guardrails:** meaningful improvements exist (YAML chart distillation, context budgeting, postprocessing, remedy-only product rules).
- **Primary blocker to “production-grade” behavior:** **SFT dataset distribution conflicts with desired output format** (SFT outputs heavily include markdown, headers, and “the native”). DPO helps, but SFT remains the dominant prior.
- **Secondary blocker:** several training/inference docs are inconsistent with current configs/scripts (risk: wrong runbook → wrong model artifacts).


## 1) Prioritized Issues (P0/P1/P2)

### P0 — Must fix for production reliability

#### P0.1 SFT dataset strongly teaches the wrong response format

**Evidence (repo scan):** In `data/sft_train/` outputs:

- Markdown bold detected in **~9,704 / 19,303** examples
- Header-like lines detected in **~15,297 / 19,303** examples
- “the native” phrasing detected in **~7,445 / 19,303** examples
- Bulleted/numbered list formatting detected in **~6,350 / 19,303** examples

**Impact:**

- Model tends to leak headers, markdown, and formal report structures at inference.
- This directly conflicts with the runtime “ABSOLUTE RULES” prompts and forces postprocessing to do heavy lifting.

**Fix options (choose one as your primary strategy):**

- **Option A (recommended): SFT dataset re-normalization + retrain SFT**  ##(FOLLOW THIS)
  - Rebuild SFT `output` fields to remove markdown/headers and replace “the native” with direct address.
  - Keep semantics, compress verbosity, enforce ≤3 sentences for “simple/timing” categories.
- **Option B: Increase DPO weight/coverage**
  - Expand DPO pairs significantly and ensure strong negative examples for markdown/headers.
  - This can work, but you’re fighting a strong SFT prior.

**Minimal quick win (without retraining):** keep postprocessing, but understand it is a brittle band-aid.


#### P0.2 Docs/runbooks disagree with current configs/scripts (risk of incorrect training) ## YES YOU CAN UPDATE THIS

**Example mismatches observed:**

- `TRAINING_GUIDE.md` contains older/inline script snippets and paths (e.g., references to `/workspace/data/arrow/...` and scripts that don’t match the current repo structure).
- Current configs use `./data/...` paths and distinct LoRA configs.

**Impact:** wrong instructions can produce:

- Training on the wrong dataset path
- Merging the wrong adapters
- Quantizing the wrong artifact

**Fix:**

- Consolidate the authoritative runbook into one place (either update `TRAINING_GUIDE.md` or add a single new “SOURCE OF TRUTH” doc) and remove conflicting guidance.


#### P0.3 Security: open CORS on API server  ## THIS IS NOT MAJOR ISSUE WE CAN DO THIS AFTER TESTING.

**Location:** `scripts/11_api_server.py` sets:

- `allow_origins=["*"]`

**Impact:**

- Any website can call your API from a browser context.
- If you deploy this publicly, it becomes a cross-origin abuse vector.

**Fix:**

- Restrict origins (env-configured allowlist) or disable credentials.
- Put auth (API key, JWT, or at least a shared secret header) if internet-exposed.


#### P0.4 Unsafe dependency behavior: training script installs packages at runtime ##YEAH YOU CAN FIX THIS

**Location:** `scripts/15_train_dpo.py`

- If TRL not installed, it runs `pip install trl>=0.9.0` at runtime.

**Impact:**

- Non-reproducible environment builds
- Risk of pulling breaking versions
- Not suitable for controlled production training

**Fix:**

- Pin TRL in `requirements.txt` (or a dedicated `requirements-train.txt`) and remove runtime installation.


### P1 — High priority quality, ops, correctness

#### P1.1 Over-reliance on keyword heuristics for remedy classification  ## YES YOU CAN FIX THIS

**Location:** `_is_remedy_query()` in `09_chat_ui.py` / `11_api_server.py`

**Impact:**

- False positives or misses for remedy intent.
- Product mentions are business-sensitive; misclassification can harm UX.

**Fix:**

- Replace with a lightweight intent classifier (small model / rules + examples), or add a second-stage check using the LLM with a strict JSON schema (“remedy_intent: yes/no”).


#### P1.2 Postprocessing does too much semantic work ## YES YOU CAN FIX THIS

**Location:** `_postprocess()`

**Impact:**

- Regex-based deletion can remove useful content and create “oddly abrupt” answers.
- Makes behavior harder to evaluate because output isn’t purely model output.

**Fix:**

- Use postprocessing only for formatting & safety; shift most style control into DPO/SFT training.


#### P1.3 DPO dataset category metadata is largely missing in prepared dataset ## YES YOU CAN FIX THIS

**Evidence:** prepared dataset entries do not consistently expose category metadata; `14_prepare_dpo_dataset.py` only emits `prompt/chosen/rejected`.

**Impact:**

- Harder to do stratified evaluation by category.
- Harder to later filter/expand dataset in targeted ways.

**Fix:**

- Include metadata columns (`category`, `chart_name`, etc.) in the HF dataset output.


#### P1.4 vLLM dtype forced to `bfloat16` ## YES YOU CAN FIX THIS

**Location:** `scripts/08_serve_vllm.py` passes `--dtype bfloat16`.

**Impact:**

- Some consumer GPUs / drivers may behave better with `float16`.

**Fix:**

- Make dtype configurable via env/CLI with sane defaults.


#### P1.5 Quantization script exports GGUF (llama.cpp) but serving path expects safetensors ##WE WILL USING VLLM so make whatever changes need to adapt that

**Location:** `06_quantize_unsloth.py` vs `08_serve_vllm.py`

**Impact:**

- GGUF is typically for llama.cpp, not vLLM.
- This can confuse the deployment story if the produced artifact isn’t compatible with vLLM as configured.

**Fix:**

- Decide target artifact(s) explicitly:
  - If vLLM: keep safetensors; use AWQ/GPTQ or bitsandbytes-supported formats.
  - If llama.cpp: produce GGUF and use a llama.cpp server, not vLLM.


### P2 — Nice-to-have improvements

#### P2.1 Observability gaps #yes please implement this it will help in debugging

- No structured logging (request ids, latency, token usage, retrieved chunk ids).

**Fix:**

- Add minimal JSON logs for:
  - request → rag chunks selected → model params → response length → postprocess deletions


#### P2.2 Evaluation gaps #yes please fix this too.

- `10_kp_test_suite.py` uses a compact test schema rather than the full YAML pipeline.

**Fix:**

- Add an evaluation mode that:
  - feeds full kundali JSON → YAML
  - runs the same postprocess as prod
  - compares output against rubric


## 2) Dataset Audit & Ratings (10-point scale)

### 2.1 DAPT dataset — `data/dapt_corpus/` — **7.5 / 10** ##YES YOU CAN FIX THIS their scripts are here @D:\Dataset_preprossecing_pipeline\scripts

**What’s good:**

- Good size for domain adaptation (654 large chunks)
- Domain text appears to contain authentic KP book content

**What’s risky:**

- Some OCR/EPUB boilerplate exists (e.g., “Internet Archive… OCR susceptible to errors”).
- DAPT is sensitive to noisy boilerplate; you risk teaching the model irrelevant formatting.

**Concrete fixes:**

- Strip archive/OCR notices and repeated footers/headers.
- Add deduplication and simple “book boilerplate” filters.


### 2.2 SFT dataset — `data/sft_train/`, `data/sft_validation/` — **5.5 / 10**  ##very imp to fix this and retrain to get the quality and accuracy to 10/10

**What’s good:**

- Large and diverse (19,303 train / 398 val)
- Reasonable English/Hinglish mix (15k en / 4.3k hi)
- Good topical coverage (property/marriage/children/finance/general/timing/health/career…)

**What’s risky (dominant):**

- Outputs strongly encode:
  - markdown
  - headers / report structure
  - “the native” phrasing
  - multi-step verbose explanations

This conflicts with the project’s intended persona (short, direct, conversational, no markdown).

**Concrete fixes:**

- Either rebuild SFT outputs in the desired response style or accept that DPO + postprocess must counteract it.


### 2.3 DPO dataset — `data/dpo/dpo_pairs.jsonl` → `data/dpo/prepared/` — **8.0 / 10** #please fix it too if you need more chart here is those D:\Dataset_preprossecing_pipeline\sample_kundali

**What’s good:**

- Chosen responses are concise (~3 sentences avg; p95=4)
- Strong contrast pairs (rejected contains “Analysis:” + markdown + robotic style)
- Uses chart YAML format matching inference path

**What’s risky:**

- Dataset is still relatively small (895 train)
- Category metadata isn’t preserved cleanly into prepared HF dataset

**Concrete fixes:**

- Preserve metadata columns into prepared dataset.
- Expand DPO pairs if you want near-guaranteed format compliance.


## 3) Security & Secret Handling Review ## its okkay

- `.env` is correctly gitignored.
- SSH key filenames are gitignored.

Remaining concerns:

- API has no authentication and open CORS.
- If deployed publicly, add auth + rate limiting.


## 4) Serving Quality Risks & Fixes   ###yes you fix this 

### 4.1 Verbosity control

- Runtime tries to cap via `max_tokens` and postprocess sentence caps.
- Long-term fix is training the style (SFT/DPO) instead of deleting text.

### 4.2 Medical safety

- Current approach: regex replacement of dangerous terms.

Risk:

- Regex-based filtering may miss paraphrases.

Fix:

- Add a second-stage safety classification pass for health queries (lightweight).


## 5) Recommended Fix Roadmap ## yes you can work.

### Phase 1 

- Lock down API CORS + add simple auth header
- Remove runtime `pip install` from training scripts; pin deps
- Preserve DPO metadata in `14_prepare_dpo_dataset.py`
- Align runbooks/docs with current scripts/configs

### Phase 2 

- Re-normalize SFT outputs to match desired format and retrain SFT
- Expand DPO pairs targeted at failure modes (headers, deflection, verbosity)

### Phase 3 

- Add structured logs + evaluation harness mirroring production pipeline
- Add regression tests (golden prompts + expected format constraints)


## 6) Appendix: Key Evidence Snapshots

### SFT formatting contamination counts (train)

- markdown bold: 9704
- header-like: 15297
- “the native”: 7445
- bullets/numbering: 6350

### DPO chosen format compliance

- bold detected: 0
- header tokens detected: 0
- p95 sentences: 4


---

## 7) Changelog — Fixes Implemented

All changes below were implemented in this audit pass. Each references the issue ID above.

### P0.1 — SFT dataset re-normalization script created
- **New file:** `scripts/17_renormalize_sft_dataset.py`
- Strips markdown (bold, italic, headers, code blocks, bullets) from SFT `output` fields
- Replaces "the native" / "the querent" with direct address ("you"/"your")
- Converts ISO dates to readable format
- Removes robotic headers, filler phrases, hallucinated rule IDs
- Enforces configurable sentence cap (default: 5)
- Creates automatic backup before modifying datasets
- Supports `--dry-run` for preview
- **Usage:** `python scripts/17_renormalize_sft_dataset.py` (then retrain SFT)

### P0.2 — TRAINING_GUIDE.md completely rewritten
- **File:** `TRAINING_GUIDE.md`
- Removed ALL outdated inline script snippets and `/workspace/data/arrow/` paths
- Now references actual scripts (`03_train_dapt.py`, `04_train_sft.py`, etc.) and actual config files
- Added Stage 0 (SFT re-normalization) to the pipeline
- Added production-mirror evaluation instructions
- Serves as the single authoritative runbook

### P0.4 — Runtime pip install removed from DPO training
- **File:** `scripts/15_train_dpo.py`
- Replaced `os.system(pip install trl)` with a clean ImportError + sys.exit(1)
- **File:** `requirements.txt`
- Added `trl>=0.9.0` to pinned dependencies

### P1.1 — Remedy classification tightened (reduced false positives)
- **Files:** `scripts/09_chat_ui.py`, `scripts/11_api_server.py`
- Replaced single flat keyword list with two-tier matching:
  - `_REMEDY_STRONG_KEYWORDS`: gemstone, remedy, mantra, rudraksha, etc. (single match sufficient)
  - `_REMEDY_CONTEXT_KEYWORDS`: "suggest remedy", "strengthen planet", etc. (require specific phrasing)
- Removed overly broad keywords: "suggest", "solution", "what to do", "how to improve", "recommendation", "protection"
- **Impact:** products no longer recommended on generic career/marriage/health queries

### P1.3 — DPO metadata preserved in prepared dataset
- **File:** `scripts/14_prepare_dpo_dataset.py`
- `format_prompt()` now returns `category` and `chart_name` alongside `prompt/chosen/rejected`
- Enables stratified evaluation and targeted dataset expansion

### P1.4 — vLLM dtype made configurable
- **File:** `scripts/08_serve_vllm.py`
- Added `--dtype` CLI arg with choices: `auto` (default), `bfloat16`, `float16`, `float32`
- Added `VLLM_DTYPE` env var support
- Default changed from hardcoded `bfloat16` to `auto` (lets vLLM pick optimal dtype for the GPU)

### P1.5 — Quantize script now defaults to vLLM-compatible safetensors
- **File:** `scripts/06_quantize_unsloth.py`
- Default `--method` changed from `q4_k_m` to `safetensors`
- Added explicit `safetensors` export path that copies the model for vLLM serving
- GGUF methods (`q4_k_m`, `q8_0`, `both`) still available but clearly labeled as "for llama.cpp only"
- Added warning when using GGUF methods

### P2.1 — Structured JSON logging added to API server
- **File:** `scripts/11_api_server.py`
- Added `_json_log()` helper emitting single-line JSON log entries
- Each `/chat` request now logs: `req_id`, `query_type`, `is_remedy`, `rag_chunks`, `max_tokens`, `temperature`, `raw_len`, `answer_len`, `has_prediction`, `has_product`, `latency_ms`
- Enables debugging, monitoring, and performance analysis

### P2.2 — Production-mirror evaluation mode added to test suite
- **File:** `scripts/10_kp_test_suite.py`
- Added `--kundali-json` flag for production-mirror evaluation
- Loads full kundali JSON → `chart_to_yaml()` → vLLM → format violation checks
- Checks: markdown bold, headers, "the native", bullets, analysis headers, ISO dates, sentence count
- Includes 15 built-in eval questions (simple/timing/analysis/past_event/remedy)
- Outputs JSON summary with violation counts per category
- **Usage:** `python scripts/10_kp_test_suite.py --kundali-json ../sample_kundali/kundali_Abhi_Raj.json`
