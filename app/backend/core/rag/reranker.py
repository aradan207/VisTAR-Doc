"""
Cross-Encoder Reranker for two-stage RAG retrieval.

Stage 1: FAISS cosine similarity retrieves a wide candidate pool (~24 chunks).
Stage 2: This module rescores each (query, chunk) pair with a fine-tuned
         cross-encoder and returns only the top_k highest-scoring chunks.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~50 MB, English-only, trained on MS-MARCO passage ranking
  - Runs on GPU automatically via sentence-transformers if CUDA is available
  - Scores are logits (raw), not probabilities — higher is more relevant
  - No new pip dependencies: sentence-transformers is already in pyproject.toml

Why this helps factual_correctness:
  Cosine similarity (Stage 1) uses independent query/chunk embeddings — it
  finds chunks that are *topically* close but may miss the exact passage that
  answers the question. The cross-encoder attends to the full (query, chunk)
  pair jointly, so it correctly ranks "injection molding pressure is defined as
  the pressure applied to the cross-sectional area" above a generic chapter
  intro from the same document.
"""

from __future__ import annotations

from typing import List, Tuple

from app.backend.core.rag.document_loader import DocumentChunk

# Lazy-loaded singleton so the model is downloaded once and cached in memory.
_reranker_instance = None


class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder for passage reranking.

    Usage:
        reranker = CrossEncoderReranker()
        top_chunks = reranker.rerank(query, faiss_results, top_k=8)
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
        # device=None → sentence-transformers picks GPU automatically if available
        self._model = CrossEncoder(self.MODEL_NAME, device=None)
        print(f"[Reranker] Loaded {self.MODEL_NAME}")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
        top_k: int = 8,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Re-score candidates with the cross-encoder and return the top_k.

        Args:
            query:      The user's search query string.
            candidates: List of (DocumentChunk, faiss_score) from Stage 1.
            top_k:      Number of results to return (default: 8).

        Returns:
            List of (DocumentChunk, cross_encoder_score), sorted descending,
            truncated to top_k.  Falls back to `candidates[:top_k]` if empty.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.content) for chunk, _ in candidates]

        # Predict scores — returns a numpy array of floats (raw logits)
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Zip chunks with new scores, sort descending, truncate
        ranked = sorted(
            zip([c for c, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]
