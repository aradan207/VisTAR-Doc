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
import subprocess
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from app.backend.core.runtime_paths import sentence_transformers_cache_available

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
# Defaults to a dedicated local HF GGUF model served through Ollama.
_DEFAULT_JUDGE = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
_JUDGE_MODEL   = os.getenv("RAGAS_JUDGE_MODEL", _DEFAULT_JUDGE)

# Sentence-transformers model for ResponseRelevancy embeddings.
# all-MiniLM-L6-v2 is tiny (~80 MB) and already used by sentence-transformers.
_EMBED_MODEL = "all-MiniLM-L6-v2"


def _resolve_embed_device() -> str:
    """Select embedding device with optional env override."""
    override = os.getenv("RAGAS_EMBED_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda"}:
        return override

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        # Fall back to CPU if torch is unavailable or CUDA probe fails.
        pass

    return "cpu"


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


def get_ragas_embed_model_name() -> str:
    """Return the sentence-transformers embedding model name used by RAGAS."""
    return _EMBED_MODEL


def _offline_mode_enabled() -> bool:
    value = os.getenv("OFFLINE_MODE", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _ollama_list_text() -> str:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except FileNotFoundError:
        return ""
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _ollama_model_present(model_name: str, listing: str) -> bool:
    if not model_name or not listing:
        return False
    return model_name.lower() in listing.lower()


def _embedding_cache_available(model_name: str) -> bool:
    return sentence_transformers_cache_available(model_name)


def validate_ragas_prerequisites() -> Dict[str, object]:
    """Validate local prerequisites for RAGAS benchmark runs.

    In OFFLINE_MODE, missing dependencies are treated as hard blockers.
    In online mode, missing embedding cache is advisory because it can be fetched.
    """
    offline_mode = _offline_mode_enabled()
    missing: List[str] = []
    warnings: List[str] = []

    ollama_listing = _ollama_list_text()
    if not ollama_listing:
        missing.append("Ollama is not available or 'ollama list' failed")
    else:
        if not _ollama_model_present(_JUDGE_MODEL, ollama_listing):
            missing.append(f"Missing RAGAS judge model in Ollama: {_JUDGE_MODEL}")
        if not _ollama_model_present(_OLLAMA_MODEL, ollama_listing):
            warnings.append(f"Agent model not found in Ollama list: {_OLLAMA_MODEL}")

    has_embed_cache = _embedding_cache_available(_EMBED_MODEL)
    if not has_embed_cache:
        message = (
            "Missing local sentence-transformers cache for "
            f"'{_EMBED_MODEL}' (first uncached run requires internet)"
        )
        if offline_mode:
            missing.append(message)
        else:
            warnings.append(message)

    ok = len(missing) == 0
    return {
        "ok": ok,
        "offline_mode": offline_mode,
        "missing": missing,
        "warnings": warnings,
        "judge_model": _JUDGE_MODEL,
        "agent_model": _OLLAMA_MODEL,
        "embed_model": _EMBED_MODEL,
    }


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

    embed_device = _resolve_embed_device()
    hf = HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL,
        model_kwargs={"device": embed_device},
    )
    _embeddings_instance = LangchainEmbeddingsWrapper(hf)
    return _embeddings_instance
