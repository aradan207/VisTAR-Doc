"""
Manual search tool for RAG-based document retrieval.
Searches the local manufacturing/machine manuals knowledge base.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from app.backend.core.agent.tool import tool
from app.backend.core.runtime_paths import (
    offline_mode_enabled,
    require_semantic_model_cache,
    sentence_transformers_cache_available,
)


class ManualSearchArgs(BaseModel):
    """Arguments for manual_search tool."""
    
    query: str = Field(
        ...,
        description="Search query to find relevant information in manufacturing/machine manuals",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of relevant document chunks to return",
    )


# Lazy-loaded singleton instances for performance
_embedder = None
_vector_store = None
_reranker = None

CANDIDATE_MULTIPLIER = 3   # First-stage FAISS over-fetches by this factor
MAX_CANDIDATES = 24        # Hard cap to keep cross-encoder latency bounded
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    """Lazy-load cross-encoder reranker (downloaded once, ~50 MB)."""
    global _reranker
    if _reranker is None:
        if offline_mode_enabled() and require_semantic_model_cache() and not sentence_transformers_cache_available(RERANKER_MODEL):
            raise RuntimeError(
                f"Offline cache missing for reranker model '{RERANKER_MODEL}'. "
                "Warm the cache on a connected machine first."
            )
        from app.backend.core.rag.reranker import CrossEncoderReranker
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_rag_components():
    """Get or initialize RAG components (lazy loading)."""
    global _embedder, _vector_store
    
    if _embedder is None:
        from app.backend.core.rag.embeddings import OllamaEmbeddings
        _embedder = OllamaEmbeddings()
    
    if _vector_store is None:
        from app.backend.core.rag.vector_store import FAISSVectorStore
        
        # Get index path from environment or use default
        project_root = Path(__file__).parent.parent.parent.parent.parent
        index_path = os.getenv(
            "INDEX_PATH",
            str(project_root / "data" / "faiss_index")
        )
        
        _vector_store = FAISSVectorStore(
            dimension=_embedder.dimension,
            index_path=index_path,
            use_gpu=True,
        )
    
    return _embedder, _vector_store


@tool(
    "manual_search",
    ManualSearchArgs,
    "**PRIMARY TOOL - USE FIRST** Search the local manufacturing/machine manuals knowledge base. "
    "This tool searches through PDF manuals containing specifications, procedures, troubleshooting guides, "
    "and technical documentation for machines like BOY injection molding machines, CNC equipment, etc. "
    "ALWAYS use this tool FIRST before web_search for ANY technical, equipment, or manufacturing question. "
    "Returns relevant excerpts with source document name and page numbers for citation."
)
def manual_search(args: ManualSearchArgs) -> dict:
    """
    Search the vector store for relevant document chunks.

    Args:
        args: ManualSearchArgs containing query and top_k

    Returns:
        dict with results list and query metadata
    """
    try:
        embedder, vector_store = _get_rag_components()
        
        # Check if index has any documents
        if vector_store.size == 0:
            return {
                "error": "No documents indexed. Run the indexer first: uv run python -m app.backend.core.rag.indexer",
                "query": args.query,
                "results": [],
            }
        
        # Generate query embedding
        query_embedding = embedder.embed(args.query)

        # --- Two-stage retrieval ---
        # Stage 1: FAISS retrieves a wider candidate pool (fast cosine similarity)
        n_candidates = min(args.top_k * CANDIDATE_MULTIPLIER, MAX_CANDIDATES)
        candidates = vector_store.search(query_embedding, top_k=n_candidates)

        # Stage 2: Cross-encoder reranker re-scores (query, chunk) pairs with a
        # fine-tuned NLI model and returns only the top_k most relevant chunks.
        # Falls back to FAISS order gracefully if the reranker fails to load.
        try:
            reranker = _get_reranker()
            results = reranker.rerank(args.query, candidates, top_k=args.top_k)
        except Exception:
            # Graceful fallback: use FAISS results directly
            results = candidates[:args.top_k]

        # Format results
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                "content": chunk.content,
                "source": chunk.source,
                "page": chunk.page,
                "relevance_score": round(score, 4),
            })
        
        return {
            "query": args.query,
            "total_documents_indexed": vector_store.size,
            "results_count": len(formatted_results),
            "results": formatted_results,
        }
        
    except FileNotFoundError:
        return {
            "error": "Index not found. Build the index first: uv run python -m app.backend.core.rag.indexer",
            "query": args.query,
            "results": [],
        }
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "query": args.query,
            "results": [],
        }


def reset_rag_components():
    """Reset RAG components (useful for testing or reloading index)."""
    global _embedder, _vector_store
    _embedder = None
    _vector_store = None
