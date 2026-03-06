Complete Pipeline Audit — KP Astrologer
CRITICAL FINDING SUMMARY
The client rated the model 2/10. After reading every script, every dataset, and running actual data quality checks, here is why — and it is worse than the GPT audit suggested.

GAP 1: SFT Dataset — The Root Cause of Everything
This is the #1 problem. The SFT dataset is fundamentally misaligned with what the client wants.

From the audit:

99% of SFT outputs are too long (>150 words) — the client wants 1-3 sentences with word  < 350 words.
22% of SFT outputs are Hinglish — the client wants English questions → English answers
99% of SFT outputs have no "ji" address — the model was never trained to say "Ritoban ji"
39% of SFT outputs use "the native" — explicitly banned by the client
26% of SFT outputs have bullet points — explicitly banned
Only 20% of SFT examples include a chart/kundali — the model was mostly trained on generic KP theory Q&A, not chart-grounded consultation
Root cause: The SFT dataset was generated as a KP textbook Q&A (e.g., "Explain what Lagna Bhava represents in KP astrology") — not as a consultation Q&A with a real user's chart. The model learned to be a KP encyclopedia, not a KP astrologer. This is why the client gets answers like "Grounding rule clearly indicate karta hai ki profession primarily public needs cater karne involve karta hai various establishments jaise: Hospitality sector..."

GAP 2: DPO Dataset — Partially Fixes the Wrong Model
From the audit:

8% of DPO pairs (205 pairs) have English questions with Hinglish chosen responses — the model is being trained to give Hinglish answers to English questions in 8% of cases
Only 8 pairs cover "past marriage" questions — client's #1 complaint ("when did I get married" → wrong answer)
Only 17 pairs cover "field of work/profession" — client's #2 complaint
Only 9 pairs cover education/graduation past events — client's #3 complaint
Only 1% of pairs cover remedy/product questions — client explicitly requested this
Only 1% cover emotional queries — model gives cold clinical responses
DPO pairs have no system_prompt field — the DPO training doesn't include the system prompt, so the model never learns to follow the persona rules during DPO
The DPO is trying to fix a fundamentally broken SFT model. You cannot DPO your way out of a bad SFT.

GAP 3: DAPT Corpus — Too Small, Wrong Format
Only 654 examples — this is tiny for domain adaptive pretraining
The sample shows OCR artifacts: "Notice This book was produced in EPUB format by the Internet Archive. The book pages were scanned and converted to EPUB format automatically."
The model is being pre-trained on OCR noise, not clean KP text
No cleaning of OCR artifacts was done before DAPT
GAP 4: RAG — Built But Not Properly Connected
From the code:

RAG is disabled by default (--no-rag is not the default but requires OPENAI_API_KEY + PINECONE_API_KEY both set)
The KB has 1,207 chunks but the category distribution is wrong: property: 352 chunks is the largest category — yet property questions are rare. career: 73 chunks is tiny for the most common question type
RAG retrieves KP book rules but the model ignores them — the system prompt doesn't tell the model how to use RAG context, it just appends it
The client's specific complaint about "field of work" getting a generic list is exactly what happens when RAG retrieves a rule like "10th house = career" and the model lists all possible careers instead of reasoning about the specific chart
GAP 5: DPO Ruleset — Missing 4 of 5 Client Requirements
The client said on Feb 21: "I don't see the rulesets in these like we had discussed. From our initial requirements: 1. Category Wise Rules. 2. Dasha Rules. 3. Planet-House Combination rules. 4. Product recommendation rules. 5. Communication Rules."

The DPO_RULESET.md contains only Communication Rules (the chosen/rejected prompt templates). It does NOT contain:

Category-wise rules (marriage, health, career, etc.) as structured JSON
Dasha interpretation rules
Planet-house combination rules
Product recommendation rules
The client provided all of these in Chat_with_client.md (Jan 30 messages) — the Venus × 12 houses framework, the planet natures, house domains, dasha examples, product examples — but none of this was converted into DPO training pairs. The DPO dataset was generated using only the communication style rules, not the actual KP knowledge rules.

GAP 6: Inference — Context Not Properly Used
From 09_chat_ui.py:

The system prompt is hardcoded with "Priya ji" examples — the model sees "Priya ji" in examples and sometimes uses it for other users (the wrong-name hallucination bug)
The chart YAML is injected but the system prompt doesn't explicitly tell the model which fields to use for which question types
Conversation history is limited to 4 turns — when the client said "but I'm already married" after asking about marriage timing, the model ignored it because context handling is weak
The post-processing has 42+ regex patterns — this is a symptom, not a solution. Each regex is a patch for a model failure that should be fixed at training time
GAP 7: Training-Inference Format Mismatch
SFT was trained with format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|> — no system prompt
Inference uses a full system prompt with chart YAML
The model was never trained to read a system prompt with chart YAML and answer questions from it
DPO training uses max_length: 1024 — but the chart YAML alone is ~14,000 chars (~3,500 tokens). The DPO training context is too short to include the full chart
GAP 8: Product Recommendations — Never Implemented
Client provided product CSV on Feb 3
12_build_product_index.py exists but the product Pinecone index (kp-products) is separate from the main KB
The DPO dataset has only 45 remedy pairs (1.8%) — the model was never properly trained to recommend products
No DPO pairs exist that show: "user asks for remedy → model recommends specific product SKU from the catalog"
GAP 9: No Evaluation Framework
The test_30q.py script was built during post-processing debugging — it tests post-processing, not model quality
There is no gold-standard test set that validates: "given this chart, does the model predict the correct dasha period?"
No regression testing between model versions
The client's 2/10 rating was the only evaluation signal
GAP 10: Missing Client-Specified Rule Categories in Training Data
The client specified these rule categories on Jan 30 (in Chat_with_client.md):

Rule Category	Client Spec	In KB	In DPO	In SFT
Marriage rules	40 rules	✅ 178 chunks	⚠️ 8%	❌ generic
Career rules	35 rules	⚠️ 73 chunks	⚠️ 6%	❌ generic
Money/Finance	30 rules	⚠️ 126 chunks	⚠️ 3%	❌ generic
Health rules	25 rules	⚠️ 79 chunks	⚠️ 3%	❌ generic
Dasha rules	HIGH priority	⚠️ 89 chunks	⚠️ 5%	❌ generic
Planet-House combos	135 entries	❌ 0 dedicated	❌ 0	❌ generic
Product recommendations	HIGH priority	❌ 0	❌ 1%	❌ 0
Communication rules	50 templates	✅ in DPO prompt	✅ partial	❌ generic
Sub-lord interpretations	150 entries	⚠️ scattered	❌ rare	❌ generic
What Needs to Be Done — Priority Order
Priority 1: Rebuild SFT Dataset (Consultation Format)
The SFT dataset must be rebuilt from scratch as chart-grounded consultation pairs — not KP textbook Q&A. Every example must include a chart YAML in the instruction and a short (1-3 sentence) response in the output. The client's Jan 30 rule structures (planet-house combos, dasha rules, category rules) must be the source of truth.

Priority 2: Build the 5 Rule JSONs the Client Asked For
Convert the client's Jan 30 specifications into proper JSON rulesets:

Category rules (marriage, career, health, finance, education, property)
Dasha interpretation rules
Planet-house combination rules (9 planets × 12 houses = 108 + special cases)
Product recommendation rules (mapped to actual SKUs from the CSV)
Communication/persona rules (already partially done in DPO_RULESET.md)
Priority 3: Rebuild DPO Dataset with Chart Context + All 5 Rule Categories
Every DPO pair must include the chart YAML in the prompt. The chosen response must demonstrate using the chart data. Cover all client-specified question types with proper distribution.

Priority 4: Fix Training-Inference Format Mismatch
SFT training must use the same format as inference: system prompt + chart YAML + user question → short answer.

Priority 5: Fix DPO max_length (1024 → 4096+)
The chart YAML is ~3,500 tokens. DPO training at 1024 tokens means the model never sees the full chart during DPO training.

Bottom line: The current model is a KP textbook that learned to recite rules in Hinglish. It was never trained to be a chart-reading consultant. The fix requires rebuilding the SFT dataset in consultation format, building the 5 rule JSONs the client asked for, and rebuilding the DPO dataset with those rules as the knowledge source. The post-processing patches in 09_chat_ui.py are band-aids on a training problem.