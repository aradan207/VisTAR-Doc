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

# The same model the agent uses (configurable via OLLAMA_MODEL in .env)
_DEFAULT_MODEL = "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"
_OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
_OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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

    from langchain_community.chat_models import ChatOllama  # noqa: F401 (community HTTP client, avoids ollama-pkg compat issues)
    from ragas.llms import LangchainLLMWrapper

    chat_model = ChatOllama(
        model=_OLLAMA_MODEL,
        base_url=_OLLAMA_HOST,
        temperature=0,          # deterministic judging
        num_predict=2048,       # must be large enough for multi-statement NLI verdicts
    )
    _llm_instance = LangchainLLMWrapper(chat_model)
    return _llm_instance


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
