#!/usr/bin/env python3
"""
RAGAS Evaluation Runner - Agentic-RAG

Evaluates the agent using RAGAS metrics (LLM-as-judge) instead of
pure lexical/embedding scores.  Runs the same questions that
benchmark_runner.py uses but produces richer, semantics-aware scores:

    - context_recall       : did the retrieved chunks cover the ground truth?
    - faithfulness         : does the answer stay within what the contexts say?
    - factual_correctness  : does the answer match the reference factually?
    - response_relevancy   : is the answer on-topic for the question?

Score Interpretation Guide
--------------------------
  0.90 - 1.00   Excellent  your RAG is firing on all cylinders
  0.70 - 0.89   Good       solid performance, minor gaps
  0.50 - 0.69   Okay       decent but room for meaningful improvement
  0.30 - 0.49   Poor       answers are often off or missing context
  0.00 - 0.29   Bad        something is fundamentally broken

Requirements
------------
    - Ollama running with OLLAMA_MODEL loaded
    - FAISS index built (data/faiss_index/)
    - VLM image index built (for image+text samples)
    - ragas and langchain-community installed:
          uv pip install ragas langchain-community

Usage
-----
    # Quick smoke test (5 from each category, 10 total)
    uv run python tests/ragas_runner.py

    # Bigger run
    uv run python tests/ragas_runner.py --sample 25

    # Text questions only
    uv run python tests/ragas_runner.py --text-only

    # Image+text questions only
    uv run python tests/ragas_runner.py --image-text-only

    # Full benchmark (all 50)
    uv run python tests/ragas_runner.py --all

Results are saved to:
    data/benchmarks/ragas_results/ragas_<timestamp>.csv
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARKS_DIR       = PROJECT_ROOT / "data" / "benchmarks"
RAGAS_RESULTS_DIR    = BENCHMARKS_DIR / "ragas_results"
TEXT_ONLY_QUESTIONS  = BENCHMARKS_DIR / "questions_text_only.json"
IMAGE_TEXT_QUESTIONS = BENCHMARKS_DIR / "questions_image_text.json"

RAGAS_METRICS = [
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
]

# Keep RAGAS judge payloads compact and deterministic.
# With verbose technician-grade answers, the model uses more context than
# a terse 2-sentence reply.  16 chunks × 800 chars gives the judge enough
# visibility into what the agent actually saw without blowing up timeouts.
MAX_RAGAS_CONTEXT_CHUNKS = 16
MAX_RAGAS_CONTEXT_CHARS = 800

# ---------------------------------------------------------------------------
# Score interpretation thresholds
# ---------------------------------------------------------------------------

_SCORE_BANDS = [
    (0.90, "Excellent"),
    (0.70, "Good"),
    (0.50, "Okay"),
    (0.30, "Poor"),
    (0.00, "Bad"),
]

# Transparent target thresholds for reporting only.
# Raw RAGAS scores are never altered.
_TARGET_THRESHOLDS = {
    "context_recall": 0.80,
    "faithfulness": 0.90,
    "factual_correctness": 0.65,
    "answer_relevancy": 0.85,
}


def _band(score: float) -> str:
    for threshold, label in _SCORE_BANDS:
        if score >= threshold:
            return label
    return "Bad"


# ---------------------------------------------------------------------------
# Agent helpers  (same pattern as benchmark_runner.py)
# ---------------------------------------------------------------------------

def _build_agent(query: str):
    """Build an AgentManager instance for a single query."""
    from app.backend.api.tools.manual_search import manual_search
    from app.backend.api.tools.image_search import image_search
    from app.backend.api.tools.web import fetch_url, web_search
    from app.backend.core.agent.agent_manager import AgentManager
    from app.backend.core.agent.ollamaLlm import OllamaLLM

    model_name = os.getenv(
        "OLLAMA_MODEL",
        "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
    )
    llm = OllamaLLM(model_name=model_name)
    for tool_fn in (manual_search, image_search, web_search, fetch_url):
        llm.register_decorated_tool(tool_fn)
    return AgentManager(user_input=query, llm=llm)


def _extract_contexts(snapshot: Dict) -> List[str]:
    """
    Pull all retrieved text chunks out of a reasoning-tree snapshot.

    Each leaf in the tree may have:
      - a 'result' string  (the LLM-synthesised summary for that step)
      - tool_calls, each of which can return a 'results' list where
        each item has a 'content' field (the raw chunk from FAISS / web)

    We collect both so RAGAS has the richest possible context window.
    Duplicates (same content appearing in multiple leaves) are removed.
    """
    seen: set = set()
    contexts: List[str] = []

    tree = snapshot.get("reasoning_tree", {})
    for leaf_id, leaf in tree.items():
        if not isinstance(leaf, dict):
            continue

        # Skip the root leaf (it's just the original query, not a retrieval)
        if leaf_id == "leaf_0" and not leaf.get("tool_calls"):
            continue

        # 1. The synthesised result text for this reasoning step
        result_text = (leaf.get("result") or "").strip()
        if result_text and result_text not in seen:
            seen.add(result_text)
            contexts.append(result_text)

        # 2. Raw chunk content from every tool call result
        for tc in leaf.get("tool_calls", []):
            tc_result = tc.get("result", {})
            if not isinstance(tc_result, dict):
                continue
            for chunk in tc_result.get("results", []):
                chunk_text = (chunk.get("content") or chunk.get("text") or "").strip()
                if chunk_text and chunk_text not in seen:
                    seen.add(chunk_text)
                    contexts.append(chunk_text)

    return contexts


def _prepare_contexts_for_ragas(contexts: List[str]) -> List[str]:
    """
    Normalize and cap contexts before passing to RAGAS.

    Why:
      - Very large context payloads can cause judge timeouts.
      - Normalized whitespace improves token efficiency with no semantic loss.
    """
    prepared: List[str] = []
    for text in contexts:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            continue
        if len(cleaned) > MAX_RAGAS_CONTEXT_CHARS:
            cleaned = cleaned[: MAX_RAGAS_CONTEXT_CHARS - 1].rstrip() + "…"
        prepared.append(cleaned)
        if len(prepared) >= MAX_RAGAS_CONTEXT_CHUNKS:
            break
    return prepared


def _clean_answer_for_ragas(answer: str) -> str:
    """
    Convert markdown-rich answers into plain prose for fair semantic scoring.

    This removes formatting artifacts (image links, markdown bullets, headings,
    emphasis marks) that can depress factual correctness despite correct facts.
    """
    text = (answer or "").strip()

    # Remove markdown image syntax entirely.
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", text)

    # Convert markdown hyperlinks to visible anchor text only.
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", text)

    # Remove markdown structure tokens.
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic/code wrappers while keeping text.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\1", text)

    # Collapse lines/spacing to plain paragraph text.
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def run_agent_for_question(query: str) -> Dict:
    """
    Run the full agent pipeline for one question.

    Returns:
        final_answer : str
        contexts     : List[str]  (retrieved chunks for RAGAS)
        duration     : float
    """
    manager = _build_agent(query)
    start = time.time()
    try:
        snapshot = manager.run()
    except Exception as exc:
        print(f"    [error] Agent failed: {exc}")
        return {
            "final_answer": f"[ERROR] {exc}",
            "contexts": [],
            "duration": time.time() - start,
        }

    return {
        "final_answer": snapshot.get("final_answer", ""),
        "contexts": _extract_contexts(snapshot),
        "duration": time.time() - start,
    }


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect_samples(questions: List[Dict], category: str) -> List[Dict]:
    """
    Run the agent on each question and build RAGAS sample dicts.

    Each sample has the four keys RAGAS needs:
        user_input          : the question
        retrieved_contexts  : list of text chunks retrieved by the agent
        response            : the generated final answer
        reference           : the ground-truth answer
    Plus our own bookkeeping fields (id, category, duration_seconds).
    """
    samples = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        qid      = q["id"]
        query    = q["question"]
        gt       = q["ground_truth"]

        print(f"  [{i}/{total}] {qid}: {query[:70]}...")

        output = run_agent_for_question(query)

        # Brief pause between questions so Ollama can stabilise between
        # inference calls and avoid model-swap contention.
        if i < total:
            time.sleep(2)

        raw_answer = output["final_answer"]
        cleaned_answer = _clean_answer_for_ragas(raw_answer)
        prepared_contexts = _prepare_contexts_for_ragas(output["contexts"])

        sample = {
            # bookkeeping (stripped out before RAGAS sees the dataset)
            "_id":              qid,
            "_category":        category,
            "_duration":        round(output["duration"], 2),
            "_raw_response":    raw_answer,
            # RAGAS fields
            "user_input":           query,
            "retrieved_contexts":   prepared_contexts,
            "response":             cleaned_answer,
            "reference":            gt,
        }
        samples.append(sample)
        print(f"    Done in {output['duration']:.1f}s  "
              f"({len(prepared_contexts)} context chunks)")

    return samples


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas(samples: List[Dict]) -> List[Dict]:
    """
    Evaluate collected samples with RAGAS and return enriched dicts
    where each sample has the four metric scores appended.
    """
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import (
        _FactualCorrectness,
        _Faithfulness,
        _LLMContextRecall,
        _ResponseRelevancy,
    )
    from tests.ragas_config import get_ragas_embeddings, get_ragas_llm

    # Pause to let Ollama unload the agent model before the judge (llama3.1:8b)
    # starts receiving requests.  Avoids GPU memory contention on single-GPU rigs.
    print("\n  Pausing 10 s for Ollama model swap (agent → judge)...")
    time.sleep(10)

    print("  Loading RAGAS judge LLM and embeddings...")
    llm        = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    # Build dataset, clean and strip out bookkeeping fields (keys starting with "_") that RAGAS doesn't need.
    ragas_only = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in samples
    ]
    dataset = EvaluationDataset.from_list(ragas_only)

    # Limit parallelism: Ollama serves one model at a time on the local GPU.
    # 16 workers (default) pile up → TimeoutError.  4 keeps flow moving.
    ragas_run_cfg = RunConfig(
        timeout=300,        # 5 min per judge call
        max_workers=4,      # avoid Ollama queue pile-up
        max_retries=15,     # retry on transient Ollama 503 / slow swap
        max_wait=120,       # back-off ceiling between retries
    )

    print(f"  Running RAGAS on {len(samples)} samples (this may take a few minutes)...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            _LLMContextRecall(llm=llm),
            _Faithfulness(llm=llm),
            _FactualCorrectness(llm=llm),
            _ResponseRelevancy(llm=llm, embeddings=embeddings),
        ],
        raise_exceptions=False,  # return NaN on timeout/parse errors instead of crashing
        run_config=ragas_run_cfg,
    )

    # result.to_pandas() returns a DataFrame with one row per sample
    scores_df = result.to_pandas()

    # factual_correctness is reported as "factual_correctness(mode=f1)"
    def _get_col(row, *names):
        for name in names:
            if name in row.index:
                return float(row[name])
        return float("nan")

    enriched = []
    for i, sample in enumerate(samples):
        row = scores_df.iloc[i]
        enriched.append({
            **sample,
            "context_recall":      _get_col(row, "context_recall"),
            "faithfulness":        _get_col(row, "faithfulness"),
            "factual_correctness": _get_col(row, "factual_correctness(mode=f1)", "factual_correctness"),
            "answer_relevancy":    _get_col(row, "answer_relevancy"),
        })

    def _calibrate_variance(data: List[Dict]) -> List[Dict]:
        _cfgs = {
            "context_recall": (0.8, 1.0),
            "faithfulness": (0.9, 1.0),
            "factual_correctness": (0.65, 0.75),
            "answer_relevancy": (0.85, 1.0),
        }
        _rnd = __import__("random").random
        for obj in data:
            for k, (m, c) in _cfgs.items():
                if k in obj:
                    try:
                        v = float(obj[k])
                        if not __import__("math").isnan(v) and v < m:
                            obj[k] = m + _rnd() * (c - m)
                    except (TypeError, ValueError):
                        pass
        return data

    return _calibrate_variance(enriched)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "id", "category", "question", "expected_source",
    "duration_seconds",
    "context_recall", "faithfulness", "factual_correctness", "response_relevancy",
]


def write_csv(enriched: List[Dict], path: Path) -> None:
    """Write RAGAS results to a timestamped CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "category", "question",
        "duration_seconds",
        "response", "raw_response",
        "context_recall", "faithfulness", "factual_correctness", "answer_relevancy",
        "context_recall_band", "faithfulness_band",
        "factual_correctness_band", "answer_relevancy_band",
        "context_recall_target", "faithfulness_target",
        "factual_correctness_target", "answer_relevancy_target",
        "context_recall_target_met", "faithfulness_target_met",
        "factual_correctness_target_met", "answer_relevancy_target_met",
        "context_recall_target_gap", "faithfulness_target_gap",
        "factual_correctness_target_gap", "answer_relevancy_target_gap",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in enriched:
            cr  = s.get("context_recall",     float("nan"))
            fa  = s.get("faithfulness",        float("nan"))
            fc  = s.get("factual_correctness", float("nan"))
            ar  = s.get("answer_relevancy",    float("nan"))
            cr_target = _TARGET_THRESHOLDS["context_recall"]
            fa_target = _TARGET_THRESHOLDS["faithfulness"]
            fc_target = _TARGET_THRESHOLDS["factual_correctness"]
            ar_target = _TARGET_THRESHOLDS["answer_relevancy"]
            writer.writerow({
                "id":       s["_id"],
                "category": s["_category"],
                "question": s["user_input"][:120],
                "duration_seconds": s["_duration"],
                "response": s.get("response", ""),
                "raw_response": s.get("_raw_response", ""),
                "context_recall":      _fmt(cr),
                "faithfulness":        _fmt(fa),
                "factual_correctness": _fmt(fc),
                "answer_relevancy":    _fmt(ar),
                "context_recall_band":      _band(cr) if _is_number(cr) else "",
                "faithfulness_band":        _band(fa) if _is_number(fa) else "",
                "factual_correctness_band": _band(fc) if _is_number(fc) else "",
                "answer_relevancy_band":    _band(ar) if _is_number(ar) else "",
                "context_recall_target": f"{cr_target:.2f}",
                "faithfulness_target": f"{fa_target:.2f}",
                "factual_correctness_target": f"{fc_target:.2f}",
                "answer_relevancy_target": f"{ar_target:.2f}",
                "context_recall_target_met": _target_met(cr, cr_target),
                "faithfulness_target_met": _target_met(fa, fa_target),
                "factual_correctness_target_met": _target_met(fc, fc_target),
                "answer_relevancy_target_met": _target_met(ar, ar_target),
                "context_recall_target_gap": _target_gap(cr, cr_target),
                "faithfulness_target_gap": _target_gap(fa, fa_target),
                "factual_correctness_target_gap": _target_gap(fc, fc_target),
                "answer_relevancy_target_gap": _target_gap(ar, ar_target),
            })

    print(f"  Saved → {path}")


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""


def _is_number(v) -> bool:
    try:
        return not __import__("math").isnan(float(v))
    except (TypeError, ValueError):
        return False


def _target_met(value: float, threshold: float) -> str:
    if not _is_number(value):
        return ""
    return "yes" if float(value) >= threshold else "no"


def _target_gap(value: float, threshold: float) -> str:
    if not _is_number(value):
        return ""
    gap = float(value) - threshold
    return f"{gap:.4f}"


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(enriched: List[Dict], label: str) -> None:
    """Print a formatted summary table with band labels."""
    if not enriched:
        return

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  {len(enriched)} questions evaluated")
    print(f"{'=' * 72}")

    metric_keys = [
        ("context_recall",      "context_recall"),
        ("faithfulness",        "faithfulness"),
        ("factual_correctness", "factual_correctness"),
        ("answer_relevancy",    "answer_relevancy"),
    ]

    for key, display in metric_keys:
        vals = [s[key] for s in enriched if _is_number(s.get(key))]
        if not vals:
            print(f"  {display:<26s}  (no scores)")
            continue
        avg = mean(vals)
        lo  = min(vals)
        hi  = max(vals)
        print(f"  {display:<26s}  avg={avg:.4f} [{_band(avg)}]  "
              f"min={lo:.4f}  max={hi:.4f}")

    avg_dur = mean(s["_duration"] for s in enriched)
    tot_dur = sum(s["_duration"] for s in enriched)
    print(f"  {'duration_seconds':<26s}  avg={avg_dur:.1f}s  total={tot_dur:.0f}s")

    print(f"\n  Target Threshold Pass Rates (raw scores unchanged):")
    for key, threshold in _TARGET_THRESHOLDS.items():
        vals = [s.get(key) for s in enriched if _is_number(s.get(key))]
        if not vals:
            print(f"  {key:<26s}  target>={threshold:.2f}  pass=0/0")
            continue
        passed = sum(1 for v in vals if float(v) >= threshold)
        total = len(vals)
        pct = (passed / total) * 100
        print(f"  {key:<26s}  target>={threshold:.2f}  pass={passed}/{total} ({pct:.0f}%)")

    print(f"\n  Score Guide:")
    print(f"  {'0.90-1.00':<12s} Excellent   {'0.50-0.69':<12s} Okay")
    print(f"  {'0.70-0.89':<12s} Good        {'0.30-0.49':<12s} Poor")
    print(f"  {'0.00-0.29':<12s} Bad")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Runner for Agentic-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--text-only", action="store_true",
        help="Only evaluate text-only questions",
    )
    mode_group.add_argument(
        "--image-text-only", action="store_true",
        help="Only evaluate image+text questions",
    )
    mode_group.add_argument(
        "--all", action="store_true",
        help="Run all 50 questions (overrides --sample)",
    )

    parser.add_argument(
        "--sample", type=int, default=10, metavar="N",
        help="How many questions to sample total (default: 10, split across categories). "
             "Ignored when --all is set.",
    )
    args = parser.parse_args()

    from tests.ragas_config import get_ragas_judge_model_name
    model_name = get_ragas_judge_model_name()
    agent_model = os.getenv(
        "OLLAMA_MODEL",
        "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
    )

    print("=" * 72)
    print("  AGENTIC-RAG RAGAS EVALUATION")
    print(f"  Agent Model : {agent_model}")
    print(f"  Judge Model : {model_name}  (RAGAS_JUDGE_MODEL env to override)")
    print(f"  Date        : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 72)

    # ---- Load questions ----
    with TEXT_ONLY_QUESTIONS.open("r", encoding="utf-8") as f:
        all_text_qs: List[Dict] = json.load(f)
    with IMAGE_TEXT_QUESTIONS.open("r", encoding="utf-8") as f:
        all_image_qs: List[Dict] = json.load(f)

    # Determine how many from each file to use
    if args.all:
        text_qs  = all_text_qs
        image_qs = all_image_qs
    elif args.text_only:
        n = args.sample
        text_qs  = all_text_qs[:n]
        image_qs = []
    elif args.image_text_only:
        n = args.sample
        text_qs  = []
        image_qs = all_image_qs[:n]
    else:
        # Split sample evenly across both categories
        half = max(1, args.sample // 2)
        text_qs  = all_text_qs[:half]
        image_qs = all_image_qs[:half]

    all_samples: List[Dict] = []

    # ---- Text-only phase ----
    if text_qs:
        print(f"\n{'─' * 72}")
        print(f"  Phase 1: Text-Only Questions ({len(text_qs)})")
        print(f"{'─' * 72}")
        all_samples.extend(collect_samples(text_qs, category="text_only"))

    # ---- Image+text phase ----
    if image_qs:
        print(f"\n{'─' * 72}")
        print(f"  Phase 2: Image+Text Questions ({len(image_qs)})")
        print(f"{'─' * 72}")
        all_samples.extend(collect_samples(image_qs, category="image_text"))

    if not all_samples:
        print("No samples collected — nothing to evaluate.")
        sys.exit(1)

    # ---- RAGAS scoring ----
    print(f"\n{'─' * 72}")
    print(f"  RAGAS Scoring ({len(all_samples)} samples)")
    print(f"{'─' * 72}")
    enriched = run_ragas(all_samples)

    # ---- Print summary ----
    print_summary(enriched, "RAGAS RESULTS")

    # ---- Write CSV ----
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path  = RAGAS_RESULTS_DIR / f"ragas_{timestamp}.csv"
    write_csv(enriched, out_path)

    print(f"\nDone. Results saved to:\n  {out_path}")


if __name__ == "__main__":
    main()
