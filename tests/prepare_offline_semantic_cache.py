#!/usr/bin/env python3
"""Prepare sentence-transformers caches required for strict offline mode."""

from __future__ import annotations


def main() -> int:
    print("[CachePrep] Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[CachePrep] all-MiniLM-L6-v2 cache ready")

    print("[CachePrep] Loading cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
    from sentence_transformers.cross_encoder import CrossEncoder

    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=None)
    print("[CachePrep] cross-encoder cache ready")

    print("[CachePrep] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
