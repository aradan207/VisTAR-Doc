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

        sample = {
            # bookkeeping (stripped out before RAGAS sees the dataset)
            "_id":              qid,
            "_category":        category,
            "_duration":        round(output["duration"], 2),
            # RAGAS fields
            "user_input":           query,
            "retrieved_contexts":   output["contexts"],
            "response":             output["final_answer"],
            "reference":            gt,
        }
        samples.append(sample)
        print(f"    Done in {output['duration']:.1f}s  "
              f"({len(output['contexts'])} context chunks)")

    return samples


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas(samples: List[Dict]) -> List[Dict]:
    """
    Evaluate collected samples with RAGAS and return enriched dicts
    where each sample has the four metric scores appended.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        _FactualCorrectness,
        _Faithfulness,
        _LLMContextRecall,
        _ResponseRelevancy,
    )
    from tests.ragas_config import get_ragas_embeddings, get_ragas_llm

    print("\n  Loading RAGAS judge LLM and embeddings...")
    llm        = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    # Build dataset — strip our private bookkeeping keys first
    ragas_only = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in samples
    ]
    dataset = EvaluationDataset.from_list(ragas_only)

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
    )

    # result.to_pandas() returns a DataFrame with one row per sample
    scores_df = result.to_pandas()

    # factual_correctness is reported as "factual_correctness(mode=f1)" in ragas 0.4
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

    return enriched


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
        "context_recall", "faithfulness", "factual_correctness", "answer_relevancy",
        "context_recall_band", "faithfulness_band",
        "factual_correctness_band", "answer_relevancy_band",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in enriched:
            cr  = s.get("context_recall",     float("nan"))
            fa  = s.get("faithfulness",        float("nan"))
            fc  = s.get("factual_correctness", float("nan"))
            ar  = s.get("answer_relevancy",    float("nan"))
            writer.writerow({
                "id":       s["_id"],
                "category": s["_category"],
                "question": s["user_input"][:120],
                "duration_seconds": s["_duration"],
                "context_recall":      _fmt(cr),
                "faithfulness":        _fmt(fa),
                "factual_correctness": _fmt(fc),
                "answer_relevancy":    _fmt(ar),
                "context_recall_band":      _band(cr) if _is_number(cr) else "",
                "faithfulness_band":        _band(fa) if _is_number(fa) else "",
                "factual_correctness_band": _band(fc) if _is_number(fc) else "",
                "answer_relevancy_band":    _band(ar) if _is_number(ar) else "",
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

    model_name = os.getenv(
        "OLLAMA_MODEL",
        "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
    )

    print("=" * 72)
    print("  AGENTIC-RAG RAGAS EVALUATION")
    print(f"  Judge Model : {model_name}")
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
