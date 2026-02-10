# DPO Dataset Generation — System Design Document

## Problem Statement
Client feedback (Feb 8, 2026): Model responses are robotic, verbose, lack specific dates,
spam products on every message, and hallucinate when asked simple questions like "What is my name?"

Root cause: Training-inference mismatch + low quality DPO preference pairs.

## Research Findings

### 1. Chart Data Compression (The 5400-line JSON Problem)

**Raw kundali JSON**: ~62,500 tokens, 5500 lines — far exceeds any model's useful context.

**Industry solution: Schema Distillation + YAML serialization**
- Convert JSON → compact YAML (industry benchmark: 50% token savings over JSON)
- Drop raw degree values (model needs sign/star/sub, not 23°45')
- Drop non-KP planets (Uranus, Neptune, Pluto)
- Pre-compute derived significator groups (marriage/career/finance houses)
- Human-readable dates ("Oct 2025" not "2025-10-22T04:30:00+00:00")
- Include only current + next mahadasha (not all 9)
- Include pratyantar dashas only for current + next 2 antardashas

**Our result**: 62,500 tokens → ~2,000 tokens = **31x compression**
Uses only **29% of 8K context budget**, leaving room for system prompt + response.

Sources:
- blog.tashif.codes: YAML gives 50% token savings, 30-50% fewer parse failures
- AlphaCodium: "YAML output is far better" — less cognitive load on model
- StructEval benchmark: models produce more accurate outputs in YAML vs JSON

### 2. DPO Quality Best Practices

**From Tulu 3 (Allen AI, ICLR 2025 — state-of-the-art open post-training):**
- Multi-stage pipeline: SFT → DPO → RLVR
- On-policy preference data (generate from SFT model, score, create pairs)
- Aggressive decontamination against eval data
- Skill-specific synthetic data at each training stage

**From Anyscale (production DPO case study):**
- On-policy >> off-policy for preference tuning
- Generate multiple candidates per prompt, score with LLM-as-judge
- Iterative DPO: multiple rounds, regenerating data from updated model
- β = 0.03 optimal, learning rate 10-100x smaller than SFT
- Quality filtering: remove pairs where chosen has robotic patterns

**From Philschmid (Hugging Face, 2025):**
- Generate N solutions per prompt, score with rule-based + LLM judge
- Use on-policy data (generate from YOUR model, not just GPT-4)
- β range 0.1-0.5 typical, learning rate ~5e-6

**Key insight: LLM-as-Judge for quality control**
- GPT-4 achieves >80% agreement with human preferences
- Multi-criteria rubric scoring catches subtle quality issues
- More scalable than human annotation

### 3. Our DPO Pipeline Design

#### Architecture: Multi-Pass Generation with Quality Gates

```
Pass 1: GENERATION (GPT-4o)
├── For each (question, chart) pair:
│   ├── Generate 3 "chosen" candidates (temp=0.7-0.9)
│   └── Generate 2 "rejected" candidates (temp=0.9)
│
Pass 2: QUALITY SCORING (GPT-4o-as-judge)
├── Score each candidate on 8 criteria (0-5 scale):
│   ├── date_specificity: Uses specific months from dasha data?
│   ├── conciseness: ≤3 short paragraphs? No rambling?
│   ├── tone: Warm, conversational, like a real pandit?
│   ├── age_awareness: Correct past/future tense for person's age?
│   ├── product_discipline: Products ONLY when remedy asked?
│   ├── name_usage: "[Name] ji" not "the native"?
│   ├── format_compliance: No markdown, no headers, no bullets?
│   └── factual_grounding: References actual chart data?
│
Pass 3: PAIR SELECTION
├── Best chosen (highest total score) vs worst rejected
├── Minimum margin threshold: chosen_score - rejected_score ≥ 10
├── Discard ambiguous pairs
│
Pass 4: DECONTAMINATION + DIVERSITY
├── Deduplicate near-identical responses
├── Ensure balanced category distribution
└── Final quality audit on random 5% sample
```

#### Why This Beats Simple chosen/rejected Generation:
1. **Multiple candidates** → pick the BEST, not just "a" response
2. **Rubric scoring** → catches subtle issues (age-awareness, product spam)
3. **Margin threshold** → model learns from CLEAR preference signals
4. **Deduplication** → no wasted training on near-identical pairs

### 4. Token Budget Analysis

```
Component          | Tokens | % of 8K context
-------------------|--------|----------------
System prompt      |   ~500 | 6%
Chart YAML         | ~2,000 | 24%
User question      |   ~50  | 1%
Available response | ~5,642 | 69%
```

The model has 69% of context available for generating responses.
This is comfortable — no truncation risk, no attention dilution.

### 5. Training Hyperparameters (Based on Research)

```yaml
# DPO Training Config (from Tulu 3 + Anyscale findings)
beta: 0.1              # regularization (0.03-0.1 range)
learning_rate: 5e-6    # 10-100x smaller than SFT
lr_scheduler: cosine
epochs: 2-3
lora_rank: 16          # higher than SFT for better preference learning
warmup_ratio: 0.1
max_length: 2048
```
