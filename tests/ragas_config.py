"""
RAGAS Configuration - LLM and Embedding wrappers for RAGAS evaluation.

Builds a judge LLM and embedding model from whatever is already configured
in the project's .env so nothing extra needs to be installed or downloaded.

LLM Judge : Ollama via ChatOllama  (langchain-community)
Embeddings: HuggingFaceEmbeddings  (sentence-transformers, already installed)

Usage
-----
from tests.ragas_config import get_ragas_llm, get_ragas_embeddings

llm        = get_ragas_llm()
embeddings = get_ragas_embeddings()
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Agent model (Ministral-8B Q4_K_M stays as the agent; configurable via OLLAMA_MODEL in .env)
_DEFAULT_MODEL = "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"
_OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
_OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Judge model: a dedicated, more capable model for RAGAS NLI evaluation.
# Uses a SEPARATE env var so it never interferes with the agent model.
# llama3.1:8b is ~4.7 GB; runs comfortably on RTX 5090 alongside the 8B agent.
_DEFAULT_JUDGE = "llama3.1:8b"
_JUDGE_MODEL   = os.getenv("RAGAS_JUDGE_MODEL", _DEFAULT_JUDGE)

# Sentence-transformers model for ResponseRelevancy embeddings.
# all-MiniLM-L6-v2 is tiny (~80 MB) and already used by sentence-transformers.
_EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Lazy-cached singletons (avoids re-loading on every import)
# ---------------------------------------------------------------------------

_llm_instance        = None
_embeddings_instance = None


def get_ragas_llm():
    """
    Return a RAGAS-compatible LLM wrapper backed by the local Ollama model.

    Lazily constructed and cached for the lifetime of the process.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    from langchain_ollama import ChatOllama  # langchain-ollama (non-deprecated)
    from ragas.llms import LangchainLLMWrapper

    chat_model = ChatOllama(
        model=_JUDGE_MODEL,
        base_url=_OLLAMA_HOST,
        temperature=0,          # deterministic judging
        num_predict=4096,       # 8B model can handle large NLI payloads without truncation
        format="json",          # grammar-constrained JSON — eliminates RagasOutputParserException
        timeout=300,            # 5 min — Ollama needs headroom when swapping 3B/8B models
    )
    _llm_instance = LangchainLLMWrapper(chat_model)
    return _llm_instance


def get_ragas_judge_model_name() -> str:
    """Return the judge model name (for display in the runner header)."""
    return _JUDGE_MODEL


def get_ragas_embeddings():
    """
    Return a RAGAS-compatible embeddings wrapper using sentence-transformers.

    The model 'all-MiniLM-L6-v2' is downloaded once on first use by
    sentence-transformers into its local cache (~80 MB).
    """
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    _embeddings_instance = LangchainEmbeddingsWrapper(hf)
    return _embeddings_instance
