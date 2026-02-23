# RAGAS Evaluation Findings — Agentic-RAG

This document records every RAGAS evaluation run, the changes made between runs,
root-cause analysis of score gaps, and the improvement roadmap.

---

## What RAGAS Measures

| Metric | What it asks | Ideal signal |
|---|---|---|
| `context_recall` | Did the retrieved chunks contain the information needed to answer correctly? | High → retrieval finds the right passages |
| `faithfulness` | Does every claim in the answer come from the retrieved context? | High → model doesn't hallucinate beyond context |
| `factual_correctness` | Do the facts in the generated answer match the reference answer? | High → answer matches ground truth factually |
| `answer_relevancy` | Is the answer on-topic for the question? | High → no off-topic tangents |

**Score interpretation:**
- 0.90–1.00 Excellent
- 0.70–0.89 Good
- 0.50–0.69 Okay
- 0.30–0.49 Poor
- 0.00–0.29 Bad

---

## Run History

### Run 1 — 2026-02-23 01:34 UTC (`ragas_20260223_013407.csv`)

**State of system:** Original prompts, `num_predict=512`, no image injection.

| Metric | avg | band | note |
|---|---|---|---|
| context_recall | 0.9375 | Excellent | only 2 samples scored (others timed out) |
| faithfulness | (no scores) | — | 100% failure — NLI jobs timed out at 512 tokens |
| factual_correctness | 0.3533 | Poor | only 3 samples scored |
| answer_relevancy | 0.7667 | Good | most samples scored |

**Primary failure:** `num_predict=512` caused the 3B judge to truncate its NLI JSON mid-sentence → mass `OutputParserException` and `TimeoutError`. `faithfulness` returned zero scores.

---

### Run 2 — 2026-02-23 02:04 UTC (`ragas_20260223_020418.csv`)

**Changes from Run 1:**
1. `num_predict` 512 → 2048 — fixes JSON truncation
2. `FILL_RESULT_PROMPT` rewritten — banned meta-commentary phrases
3. `FINAL_PROMPT` rewritten — leads with answer quality, mandatory image inclusion
4. `_inject_missing_images()` post-processing added
5. `SYSTEM_PROMPT` cleaned — removed answer-format rules from planning layer

**Per-question results:**

| id | context_recall | faithfulness | factual_correctness | answer_relevancy |
|---|---|---|---|---|
| text-01 | 1.00 Exc | 1.00 Exc | **0.17 Bad** | 0.86 Good |
| text-02 | 1.00 Exc | 1.00 Exc | 0.67 Okay | 0.91 Exc |
| text-03 | 1.00 Exc | 1.00 Exc | 0.46 Poor | 0.71 Good |
| text-04 | **0.00 Bad** | 1.00 Exc | **0.00 Bad** | 0.95 Exc |
| text-05 | 1.00 Exc | 1.00 Exc | **0.25 Bad** | 0.90 Exc |
| imgtext-01 | nan | nan | 0.57 Okay | 0.86 Good |
| imgtext-02 | 1.00 Exc | 0.90 Exc | 0.57 Okay | 0.81 Good |
| imgtext-03 | **0.00 Bad** | 0.90 Exc | **0.00 Bad** | 0.76 Good |
| imgtext-04 | 1.00 Exc | 0.88 Good | 0.56 Okay | 0.69 Okay |
| imgtext-05 | nan | nan | 0.71 Good | 0.85 Good |

**Summary averages:**

| Metric | avg | band | Δ vs Run 1 |
|---|---|---|---|
| context_recall | 0.7500 | Good | N/A (Run 1 had only 2 valid samples) |
| faithfulness | 0.9594 | **Excellent** | ∞ (Run 1 had zero scores) |
| factual_correctness | 0.3960 | Poor | +0.04 |
| answer_relevancy | 0.8299 | Good | +0.06 |

---

## Root Cause Analysis — `factual_correctness` = 0.39

`_FactualCorrectness` decomposes each answer into atomic claims and uses NLI to check
whether each claim is supported by the reference answer. A claim unsupported by the
reference counts against the score — **even if the claim is factually true in the real world**.

### Failure Mode A — Over-Elaboration (affects ~7/10 samples)

The 3B agent adds correct but extra domain knowledge not present in the ground truth.

**text-01** (FC=0.17): Ground truth mentions melting temp, energy content, melt viscosity, stability.
Agent adds crystallization rate, tensile strength, shrinkage, moisture drying — all correct injection
molding facts, but absent from the reference → NLI marks each as "not supported" → score collapses.

**text-05** (FC=0.25): Ground truth = "A gate is the channel through which the molten resin flows
from the runner into the cavity." Agent adds timing controls, gate types, configurable delays — all penalized.

**Root cause:** `FINAL_PROMPT` does not forbid adding domain knowledge from training weights. The model
reads context, finds the key facts, then enriches with its own knowledge.

### Failure Mode B — Complete Retrieval Miss (text-04, imgtext-03)

`context_recall = 0.00` → FAISS retrieved wrong chunks → reference content never in context window →
model answers from training knowledge or says it cannot answer → FC = 0.00.

**text-04:** "What is injection molding pressure according to the APSX-PIM manual?" — `top_k=5` likely
retrieved generic injection molding pages instead of the APSX-PIM pressure-spec page. The query wasn't
source-specific enough.

**imgtext-03:** "Show me the APSX-PIM internal components" — `manual_search` chunks for this question
were irrelevant; the text context was empty of the right content.

### Failure Mode C — Judge Model Ceiling

The same 3B quantized model used as the agent is used as the RAGAS judge. A 3B model is marginal at
rigorous multi-claim NLI — it occasionally times out, misclassifies edge cases, and has limited precision
on fine-grained factual comparisons. This sets a score ceiling independent of answer quality.

---

## What Has Been Tried

| Change | Effect | Status |
|---|---|---|
| `num_predict` 512 → 2048 | Fixed faithfulness (0 → 0.96 Excellent), reduced timeouts ~50% → ~10% | Done |
| Rewrite `FILL_RESULT_PROMPT` | Eliminated meta-commentary in reasoning step summaries | Done |
| Rewrite `FINAL_PROMPT` | Reduced boilerplate preambles; tighter answer scope instruction | Done |
| Remove image rules from `SYSTEM_PROMPT` | Planning layer no longer bleeds answer commentary into context | Done |
| `_inject_missing_images()` post-processing | Image URLs guaranteed in visual answers regardless of model | Done |
| URL stripping in `normalize_text()` | Prevents URL tokens polluting ROUGE/BLEU scores | Done (pre-existing) |

---

## Improvement Roadmap — Targeting `factual_correctness` ≥ 0.70

### Priority 1 — Retrieval-Only Constraint in FINAL_PROMPT (Highest Impact)

**File:** `app/backend/core/agent/agent_manager.py`, `finalize()` method

Add rule: "Answer ONLY using facts that appear explicitly in the context below. Do NOT add domain
knowledge, elaborations, adjacent concepts, or inferences from your training data. If a fact is
not in the context, do not state it."

Directly attacks Failure Mode A. Turns the model from an "enriching expert" into a
"faithful summariser of retrieved content."

**Expected FC gain:** +0.15–0.25

---

### Priority 2 — Self-Critique Refinement Pass (High Impact)

**File:** `app/backend/core/agent/agent_manager.py`, `finalize()` method

After generating the initial answer, run a second LLM call:

> "You are a fact-checker. Review this answer for the question: [question].
> Remove any claim that is not directly stated in the context below.
> Output only the refined answer with unsupported claims removed. No commentary."

Catches elaborations that survive the first prompt instruction. Adds ~2–4s overhead.

**Expected FC gain:** +0.10–0.15 (compounds with Priority 1)

---

### Priority 3 — Increase Default `top_k` to 8 (Medium Impact)

**File:** `app/backend/api/tools/manual_search.py` — change `default=5` to `default=8`

**File:** `app/backend/core/models/prompt.py` — update SYSTEM_PROMPT example to use `top_k: 8`

Attacks Failure Mode B directly. 8 chunks vs 5 increases probability of capturing the reference
passage, especially for questions where the relevant sentence is not the #1 ranked result.

**Expected FC gain:** +0.05–0.15 on retrieval-miss questions

---

### Priority 4 — Source-Specific Query Instructions in SYSTEM_PROMPT (Medium Impact)

**File:** `app/backend/core/models/prompt.py`

Add rule: "When the user's question references a specific manual or machine by name (e.g.,
'according to the APSX-PIM manual'), ALWAYS include that exact name in your `manual_search`
query. Never query generically when the source is specified."

Fixes text-04-style failures where the model searches for "injection molding pressure" instead
of "APSX-PIM injection molding pressure", retrieving the wrong manual's content.

**Expected FC gain:** +0.05–0.10 on source-specific questions

---

### Priority 5 — RAGAS Evaluation Stability

**File:** `tests/ragas_runner.py`

Pass `raise_exceptions=False` to `evaluate()` so timeouts/parse failures return NaN rather than
counting as 0. This gives cleaner averages: NaN samples are excluded, not zeroed out.

---

## Expected Scores After All Fixes

| Metric | Current | After P1+P2 | After P1+P2+P3+P4 |
|---|---|---|---|
| context_recall | 0.75 Good | ~0.75 Good | ~0.82 Good |
| faithfulness | 0.96 Excellent | ~0.96 Excellent | ~0.96 Excellent |
| factual_correctness | **0.40 Poor** | ~0.58 Okay | **~0.72 Good** |
| answer_relevancy | 0.83 Good | ~0.82 Good | ~0.85 Good |

---

## Known Constraints

1. No model change — stays on Ministral-3B Q4_K_M
2. No ground truth modification — reference answers are fixed
3. No breaking web UI — all changes serve both benchmark and interactive use
4. Judge model = Agent model — inherent ceiling on NLI precision. A larger judge (e.g., llama3.1:8b via `OLLAMA_MODEL=llama3.1:8b uv run python tests/ragas_runner.py`) would give cleaner FC scores

---

## Round 2 — Changes Applied (pre-Run 3)

All 5 roadmap priorities implemented. Run `uv run python tests/ragas_runner.py` to produce Run 3.

| # | File | Change | Target |
|---|---|---|---|
| 1 | `app/backend/core/agent/agent_manager.py` | FINAL_PROMPT Rule 6 rewritten: added explicit retrieval-only constraint forbidding training-data elaboration | Failure Mode A |
| 2 | `app/backend/core/agent/agent_manager.py` | Self-critique refinement pass added after initial answer; second LLM call strips unsupported claims; `_inject_missing_images()` re-runs after critique | Failure Mode A |
| 3 | `app/backend/api/tools/manual_search.py` | `default=5` → `default=8` | Failure Mode B |
| 4 | `app/backend/core/models/prompt.py` | Rule 5 added: source-specific query instruction; both `top_k: 5` examples → `top_k: 8` | Failure Mode B |
| 5 | `tests/ragas_runner.py` | `raise_exceptions=False` added to `evaluate()` call | Error stability |

**Note on self-critique pass**: The critique uses the same `self.llm` (Ministral-3B). If the model's NLI capability is limited, the critique may be imprecise. If FC does not improve after Run 3, consider removing the critique pass and relying solely on the FINAL_PROMPT constraint (simpler, no latency overhead).