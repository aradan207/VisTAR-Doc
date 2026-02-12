#!/usr/bin/env python3
"""
Benchmark Runner - Agentic-RAG Evaluation Pipeline

This script runs the full benchmark evaluation for the Agentic-RAG system.
It loads 50 ground truth questions (25 text-only, 25 image+text), sends each
through the agent pipeline, scores the generated answers against the reference
answers, and writes structured CSV results to data/benchmarks/.

Requirements:
    - Ollama must be running with the configured model loaded
    - The FAISS index must be built (data/faiss_index/)
    - The VLM image index must exist (for image+text questions)

Usage:
    uv run python tests/benchmark_runner.py
    uv run python tests/benchmark_runner.py --text-only
    uv run python tests/benchmark_runner.py --image-text-only
"""

import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from tests.benchmark_metrics import (
    check_image_presence,
    compute_all_metrics,
    extract_image_urls,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"
TEXT_ONLY_QUESTIONS = BENCHMARKS_DIR / "questions_text_only.json"
IMAGE_TEXT_QUESTIONS = BENCHMARKS_DIR / "questions_image_text.json"

RESULTS_TEXT_ONLY_CSV = BENCHMARKS_DIR / "results_text_only.csv"
RESULTS_IMAGE_TEXT_CSV = BENCHMARKS_DIR / "results_image_text.csv"
RESULTS_ALL_CSV = BENCHMARKS_DIR / "results_all.csv"

METRIC_FIELDS = [
    "token_f1",
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "bertscore_f1",
    "retrieval_accuracy",
]

IMAGE_TEXT_EXTRA_FIELDS = [
    "image_present",
    "image_url_correct_source",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A single benchmark evaluation record."""
    id: str
    category: str  # "text_only" or "image_text"
    question: str
    expected_source: str
    ground_truth: str
    generated_answer: str
    retrieved_sources: str  # comma-separated list of sources found
    duration_seconds: float
    token_f1: float
    bleu: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    bertscore_f1: float
    retrieval_accuracy: float
    image_present: Optional[bool] = None
    image_url_correct_source: Optional[bool] = None


# ---------------------------------------------------------------------------
# Agent invocation
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


def run_agent_for_question(query: str) -> Dict:
    """
    Run the full agent pipeline for a single question.

    Returns a dict with:
        - final_answer: the generated text
        - retrieved_sources: list of source PDFs found during tool calls
        - duration: wall-clock seconds
    """
    manager = _build_agent(query)

    start = time.time()
    try:
        result = manager.run()
    except Exception as exc:
        print(f"    [error] Agent failed: {exc}")
        return {
            "final_answer": f"[ERROR] {exc}",
            "retrieved_sources": [],
            "duration": time.time() - start,
        }
    elapsed = time.time() - start

    final_answer = result.get("final_answer", "")

    # Extract sources from the reasoning tree.
    # reasoning_tree.to_dict() returns a *flat* dict: {"leaf_0": {...}, "leaf_1": {...}, ...}
    sources: List[str] = []
    tree = result.get("reasoning_tree", {})
    for leaf in tree.values():
        if not isinstance(leaf, dict):
            continue
        for tc in leaf.get("tool_calls", []):
            tc_result = tc.get("result", {})
            if isinstance(tc_result, dict):
                for r in tc_result.get("results", []):
                    src = r.get("source") or r.get("pdf_source") or ""
                    if src and src not in sources:
                        sources.append(src)

    # Fallback: extract source names mentioned in the generated answer text
    # (e.g. "APSX-PIM.pdf, page 3" or "BOY-35" references).
    if not sources and final_answer:
        _SOURCE_RE = re.compile(
            r'\b(APSX-PIM|BOY-35|MILACRON|MAAC|RSA-G2|DSC|FTIR'
            r'|Manual_BOY-35[^,)\s]*|Manual_MAAC[^,)\s]*)'
            r'(?:\.pdf)?\b',
            re.IGNORECASE,
        )
        for m in _SOURCE_RE.finditer(final_answer):
            name = m.group(1)
            # Normalise common aliases
            normalised = name.upper().replace("MANUAL_", "").split(".")[0]
            if normalised not in sources:
                sources.append(normalised)

    return {
        "final_answer": final_answer,
        "retrieved_sources": sources,
        "duration": elapsed,
    }


# ---------------------------------------------------------------------------
# Pre-scoring cleanup
# ---------------------------------------------------------------------------

def _clean_for_scoring(text: str) -> str:
    """Strip image markdown and raw URLs so they don't corrupt NLG metrics.

    The raw answer is still saved in the CSV for manual inspection; this
    cleaned version is only used when computing token-overlap scores.
    """
    # Remove image markdown: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove raw URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse extra whitespace left behind
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def evaluate_text_only(questions: List[Dict]) -> List[BenchmarkResult]:
    """Evaluate text-only questions (no images expected in output)."""
    results: List[BenchmarkResult] = []

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["question"]
        expected_source = q["expected_source"]
        ground_truth = q["ground_truth"]

        print(f"\n  [{i}/{len(questions)}] {qid}: {query[:60]}...")

        agent_output = run_agent_for_question(query)
        generated = agent_output["final_answer"]
        retrieved = agent_output["retrieved_sources"]
        duration = agent_output["duration"]

        # Score (clean text so image markdown / URLs don't skew metrics)
        cleaned = _clean_for_scoring(generated)
        metrics = compute_all_metrics(
            ground_truth=ground_truth,
            generated=cleaned,
            expected_source=expected_source,
            retrieved_sources=retrieved,
        )

        # Check that no images leaked into a text-only answer
        has_image = check_image_presence(generated)
        if has_image:
            print(f"    [note] Image markdown found in text-only answer")

        record = BenchmarkResult(
            id=qid,
            category="text_only",
            question=query,
            expected_source=expected_source,
            ground_truth=ground_truth,
            generated_answer=generated,
            retrieved_sources=", ".join(retrieved),
            duration_seconds=round(duration, 2),
            image_present=has_image,
            image_url_correct_source=None,
            **metrics,
        )
        results.append(record)

        src_label = "Source=MATCH" if metrics['retrieval_accuracy'] == 1.0 else "Source=MISMATCH"
        print(f"    F1={metrics['token_f1']:.3f}  BLEU={metrics['bleu']:.3f}  "
              f"ROUGE-L={metrics['rougeL_f1']:.3f}  BERT={metrics['bertscore_f1']:.3f}  "
              f"{src_label}  ({duration:.1f}s)")

    return results


def evaluate_image_text(questions: List[Dict]) -> List[BenchmarkResult]:
    """Evaluate image+text questions (images expected in output)."""
    results: List[BenchmarkResult] = []

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["question"]
        expected_source = q["expected_source"]
        expected_image_pattern = q.get("expected_image_pattern", "")
        ground_truth = q["ground_truth"]

        print(f"\n  [{i}/{len(questions)}] {qid}: {query[:60]}...")

        agent_output = run_agent_for_question(query)
        generated = agent_output["final_answer"]
        retrieved = agent_output["retrieved_sources"]
        duration = agent_output["duration"]

        # Score text quality (clean text so image markdown / URLs don't skew metrics)
        cleaned = _clean_for_scoring(generated)
        metrics = compute_all_metrics(
            ground_truth=ground_truth,
            generated=cleaned,
            expected_source=expected_source,
            retrieved_sources=retrieved,
        )

        # Check image presence and source correctness
        has_image = check_image_presence(generated)
        image_urls = extract_image_urls(generated)
        image_source_correct = None
        if has_image and image_urls:
            # Check if any URL contains the expected source pattern
            image_source_correct = any(
                expected_source.lower() in url.lower()
                for url in image_urls
            )

        record = BenchmarkResult(
            id=qid,
            category="image_text",
            question=query,
            expected_source=expected_source,
            ground_truth=ground_truth,
            generated_answer=generated,
            retrieved_sources=", ".join(retrieved),
            duration_seconds=round(duration, 2),
            image_present=has_image,
            image_url_correct_source=image_source_correct,
            **metrics,
        )
        results.append(record)

        img_status = "IMG_CORRECT" if has_image else "NO_IMG"
        src_label = "Source=MATCH" if metrics['retrieval_accuracy'] == 1.0 else "Source=MISMATCH"
        print(f"    F1={metrics['token_f1']:.3f}  BLEU={metrics['bleu']:.3f}  "
              f"ROUGE-L={metrics['rougeL_f1']:.3f}  BERT={metrics['bertscore_f1']:.3f}  "
              f"{src_label}  {img_status}  ({duration:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _csv_fieldnames() -> List[str]:
    """Return the ordered list of CSV column names."""
    return [f.name for f in fields(BenchmarkResult)]


def write_csv(records: List[BenchmarkResult], path: Path) -> None:
    """Write benchmark records to a CSV file with an aggregate summary row."""
    if not records:
        print(f"  No records to write to {path.name}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = asdict(record)
            # Truncate long text fields for CSV readability
            row["ground_truth"] = row["ground_truth"][:300]
            row["generated_answer"] = row["generated_answer"][:500]
            writer.writerow(row)

        # Write aggregate summary row
        summary = {fname: "" for fname in fieldnames}
        summary["id"] = "AVERAGE"
        summary["category"] = records[0].category
        summary["question"] = f"Aggregate over {len(records)} questions"

        for metric in METRIC_FIELDS:
            values = [getattr(r, metric) for r in records]
            summary[metric] = round(mean(values), 4)

        summary["duration_seconds"] = round(
            mean(r.duration_seconds for r in records), 2
        )

        # Image stats for image_text category
        if records[0].category == "image_text":
            img_present_count = sum(1 for r in records if r.image_present)
            img_correct_count = sum(1 for r in records if r.image_url_correct_source)
            summary["image_present"] = f"{img_present_count}/{len(records)}"
            summary["image_url_correct_source"] = f"{img_correct_count}/{len(records)}"

        writer.writerow(summary)

    print(f"  Wrote {len(records)} records + summary to {path.name}")


def print_summary(records: List[BenchmarkResult], label: str) -> None:
    """Print a formatted summary table to the console."""
    if not records:
        return

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  {len(records)} questions evaluated")
    print(f"{'=' * 72}")

    # Map internal field names to human-readable labels for the summary
    _display_labels = {"retrieval_accuracy": "source_accuracy"}
    for metric in METRIC_FIELDS:
        values = [getattr(r, metric) for r in records]
        avg = mean(values)
        lo = min(values)
        hi = max(values)
        label = _display_labels.get(metric, metric)
        print(f"  {label:<22s}  avg={avg:.4f}  min={lo:.4f}  max={hi:.4f}")

    avg_time = mean(r.duration_seconds for r in records)
    total_time = sum(r.duration_seconds for r in records)
    print(f"  {'duration_seconds':<22s}  avg={avg_time:.1f}s  total={total_time:.1f}s")

    if records[0].category == "image_text":
        img_present = sum(1 for r in records if r.image_present)
        img_correct = sum(1 for r in records if r.image_url_correct_source)
        print(f"  {'image_present':<22s}  {img_present}/{len(records)}")
        print(f"  {'image_source_correct':<22s}  {img_correct}/{len(records)}")

    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agentic-RAG Benchmark Runner")
    parser.add_argument(
        "--text-only", action="store_true",
        help="Run only text-only evaluation",
    )
    parser.add_argument(
        "--image-text-only", action="store_true",
        help="Run only image+text evaluation",
    )
    args = parser.parse_args()

    run_text = not args.image_text_only
    run_image = not args.text_only

    model_name = os.getenv(
        "OLLAMA_MODEL",
        "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
    )

    print("=" * 72)
    print("  AGENTIC-RAG BENCHMARK EVALUATION")
    print(f"  Model: {model_name}")
    print(f"  Date:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 72)

    all_records: List[BenchmarkResult] = []

    # --- Text-only evaluation ---
    if run_text:
        print("\n" + "-" * 72)
        print("  PHASE 1: Text-Only Questions (25)")
        print("-" * 72)

        with TEXT_ONLY_QUESTIONS.open("r", encoding="utf-8") as f:
            text_questions = json.load(f)

        text_results = evaluate_text_only(text_questions)
        write_csv(text_results, RESULTS_TEXT_ONLY_CSV)
        print_summary(text_results, "TEXT-ONLY RESULTS")
        all_records.extend(text_results)

    # --- Image+text evaluation ---
    if run_image:
        print("\n" + "-" * 72)
        print("  PHASE 2: Image + Text Questions (25)")
        print("-" * 72)

        with IMAGE_TEXT_QUESTIONS.open("r", encoding="utf-8") as f:
            image_questions = json.load(f)

        image_results = evaluate_image_text(image_questions)
        write_csv(image_results, RESULTS_IMAGE_TEXT_CSV)
        print_summary(image_results, "IMAGE+TEXT RESULTS")
        all_records.extend(image_results)

    # --- Combined CSV ---
    if all_records:
        write_csv(all_records, RESULTS_ALL_CSV)
        print_summary(all_records, "COMBINED RESULTS (ALL 50 QUESTIONS)")

    print("\nBenchmark complete. Results saved to:")
    if run_text:
        print(f"  {RESULTS_TEXT_ONLY_CSV}")
    if run_image:
        print(f"  {RESULTS_IMAGE_TEXT_CSV}")
    if all_records:
        print(f"  {RESULTS_ALL_CSV}")


if __name__ == "__main__":
    main()
