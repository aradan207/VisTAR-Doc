#!/usr/bin/env python3
"""
Benchmark Metrics - NLG Scoring Functions for RAG Evaluation

This module provides the metric computation functions used by the benchmark
runner to compare generated answers against ground truth references.

Metrics implemented:
    - Token F1 (SQuAD-style word overlap)
    - BLEU-4 (n-gram precision with brevity penalty)
    - ROUGE-1 F1 (unigram overlap)
    - ROUGE-2 F1 (bigram overlap)
    - ROUGE-L F1 (longest common subsequence)
    - BERTScore F1 (contextual embedding similarity)
    - Retrieval Accuracy (binary: correct source retrieved)
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Text normalization (shared by Token F1 and BLEU)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip URLs, strip articles, remove punctuation, collapse whitespace."""
    # Strip URLs before punctuation removal so path components don't become junk tokens
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"localhost\S+", "", text)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    """Normalize then split into tokens."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


# ---------------------------------------------------------------------------
# Token F1 (SQuAD-style)
# ---------------------------------------------------------------------------

def calculate_token_f1(ground_truth: str, generated: str) -> float:
    """
    Token-level F1 between ground truth and generated answer.

    This is the standard metric used in the SQuAD reading-comprehension
    benchmark.  It computes precision and recall over the bag of tokens
    after normalization, then returns their harmonic mean.
    """
    gt_tokens = tokenize(ground_truth)
    gen_tokens = tokenize(generated)

    if not gt_tokens or not gen_tokens:
        return 0.0

    common = Counter(gt_tokens) & Counter(gen_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(gen_tokens)
    recall = num_common / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU-4 (with smoothing)
# ---------------------------------------------------------------------------

def calculate_bleu(ground_truth: str, generated: str) -> float:
    """
    Smoothed BLEU-4 score.

    Uses add-one smoothing on n-gram counts and a brevity penalty to
    handle short hypotheses.  The maximum n-gram order is capped at 4 or
    the length of the shorter sequence, whichever is less.
    """
    ref_tokens = tokenize(ground_truth)
    hyp_tokens = tokenize(generated)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    max_n = min(4, len(ref_tokens), len(hyp_tokens))
    if max_n == 0:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = max(len(hyp_tokens) - n + 1, 0)
        if total == 0:
            continue
        # add-one smoothing
        precisions.append((overlap + 1) / (total + 1))

    if not precisions:
        return 0.0

    log_precision = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_precision / len(precisions))

    # brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return bp * geo_mean


# ---------------------------------------------------------------------------
# ROUGE (1, 2, L) -- delegates to the rouge-score library
# ---------------------------------------------------------------------------

# Lazy-loaded scorer to avoid import cost when not needed
_rouge_scorer = None


def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def calculate_rouge(ground_truth: str, generated: str) -> Dict[str, float]:
    """
    Returns a dict with keys rouge1_f1, rouge2_f1, rougeL_f1.
    """
    scorer = _get_rouge_scorer()
    scores = scorer.score(ground_truth, generated)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


# ---------------------------------------------------------------------------
# BERTScore F1
# ---------------------------------------------------------------------------

_bert_device = None


def _get_bert_device() -> str:
    global _bert_device
    if _bert_device is None:
        try:
            import torch
            _bert_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _bert_device = "cpu"
    return _bert_device


def calculate_bertscore(ground_truth: str, generated: str) -> float:
    """
    BERTScore F1 using bert-base-uncased.

    BERTScore computes token-level cosine similarities between contextual
    embeddings of the reference and candidate sentences, then aggregates
    them into precision, recall, and F1.
    """
    from bert_score import score as bert_score_fn

    device = _get_bert_device()
    try:
        _P, _R, F1 = bert_score_fn(
            [generated],
            [ground_truth],
            lang="en",
            model_type="bert-base-uncased",
            device=device,
            verbose=False,
        )
        return F1.mean().item()
    except Exception as e:
        print(f"[warning] BERTScore computation failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Retrieval accuracy (binary)
# ---------------------------------------------------------------------------

def calculate_retrieval_accuracy(
    expected_source: str, retrieved_sources: List[str]
) -> float:
    """
    Returns 1.0 if the expected source appears in the retrieved sources,
    0.0 otherwise.  Matching is case-insensitive substring.
    """
    expected_lower = expected_source.lower()
    for source in retrieved_sources:
        if expected_lower in source.lower():
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Image presence check (for image+text benchmarks)
# ---------------------------------------------------------------------------

_IMAGE_MARKDOWN_RE = re.compile(r"!\[.*?\]\(.*?\)")


def check_image_presence(generated: str) -> bool:
    """Returns True if the generated answer contains image markdown."""
    return bool(_IMAGE_MARKDOWN_RE.search(generated))


def extract_image_urls(generated: str) -> List[str]:
    """Extract all image URLs from markdown image syntax in the answer."""
    return re.findall(r"!\[.*?\]\((.*?)\)", generated)


# ---------------------------------------------------------------------------
# Full scoring pipeline
# ---------------------------------------------------------------------------

def compute_all_metrics(
    ground_truth: str,
    generated: str,
    expected_source: str,
    retrieved_sources: List[str],
) -> Dict[str, float]:
    """
    Compute every metric for a single (ground_truth, generated) pair.

    Returns a dict with keys:
        token_f1, bleu, rouge1_f1, rouge2_f1, rougeL_f1,
        bertscore_f1, retrieval_accuracy
    """
    rouge = calculate_rouge(ground_truth, generated)

    return {
        "token_f1": round(calculate_token_f1(ground_truth, generated), 4),
        "bleu": round(calculate_bleu(ground_truth, generated), 4),
        "rouge1_f1": round(rouge["rouge1_f1"], 4),
        "rouge2_f1": round(rouge["rouge2_f1"], 4),
        "rougeL_f1": round(rouge["rougeL_f1"], 4),
        "bertscore_f1": round(calculate_bertscore(ground_truth, generated), 4),
        "retrieval_accuracy": calculate_retrieval_accuracy(
            expected_source, retrieved_sources
        ),
    }
