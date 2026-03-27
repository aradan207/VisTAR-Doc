#!/usr/bin/env python3
"""Offline readiness checker for strict image-capable deployments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.core.runtime_paths import (
    offline_mode_enabled,
    require_image_retrieval,
    require_semantic_model_cache,
    required_image_artifact_paths,
    resolve_yologen_processed_dir,
    sentence_transformers_cache_available,
)


def _project_root() -> Path:
    return PROJECT_ROOT


def _ollama_list() -> str:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False, timeout=20)
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _check_ollama_models() -> Dict[str, object]:
    required = [
        os.getenv("OLLAMA_MODEL", "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"),
        os.getenv("OLLAMA_EMBED_MODEL", "hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M"),
    ]

    listing = _ollama_list()
    missing = []
    if not listing:
        return {"ok": False, "missing": ["ollama unavailable or 'ollama list' failed"], "required": required}

    lower = listing.lower()
    for model in required:
        if model.lower() not in lower:
            missing.append(model)

    return {"ok": len(missing) == 0, "missing": missing, "required": required}


def _check_faiss(project_root: Path) -> Dict[str, object]:
    index_root = Path(os.getenv("INDEX_PATH", str(project_root / "data" / "faiss_index")))
    required = [index_root / "index.faiss", index_root / "chunks.json"]
    missing = [str(p) for p in required if not p.exists()]
    return {"ok": len(missing) == 0, "index_root": str(index_root), "missing": missing}


def _check_image_artifacts(project_root: Path) -> Dict[str, object]:
    processed = resolve_yologen_processed_dir(project_root)
    if processed is None:
        return {
            "ok": False,
            "processed_root": None,
            "missing": ["yologen processed root not found"],
        }

    required = required_image_artifact_paths(processed)
    missing = [str(p) for p in required if not p.exists()]
    return {
        "ok": len(missing) == 0,
        "processed_root": str(processed),
        "missing": missing,
    }


def _check_semantic_cache() -> Dict[str, object]:
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    missing = [m for m in models if not sentence_transformers_cache_available(m)]
    return {"ok": len(missing) == 0, "missing": missing, "models": models}


def _strict_mode_result(report: Dict[str, object]) -> Dict[str, object]:
    blockers: List[str] = []

    if not report["checks"]["faiss"]["ok"]:
        blockers.append("faiss index missing")

    if not report["checks"]["ollama"]["ok"]:
        blockers.append("ollama models unavailable")

    if report["flags"]["require_image_retrieval"] and not report["checks"]["image_artifacts"]["ok"]:
        blockers.append("image artifacts missing")

    if report["flags"]["require_semantic_model_cache"] and not report["checks"]["semantic_cache"]["ok"]:
        blockers.append("semantic model cache missing")

    report["ready"] = len(blockers) == 0
    report["blockers"] = blockers
    return report


def main() -> int:
    project_root = _project_root()

    report: Dict[str, object] = {
        "project_root": str(project_root),
        "flags": {
            "offline_mode": offline_mode_enabled(),
            "require_image_retrieval": require_image_retrieval(),
            "require_semantic_model_cache": require_semantic_model_cache(),
        },
        "checks": {
            "faiss": _check_faiss(project_root),
            "image_artifacts": _check_image_artifacts(project_root),
            "semantic_cache": _check_semantic_cache(),
            "ollama": _check_ollama_models(),
        },
    }

    report = _strict_mode_result(report)

    print(json.dumps(report, indent=2))
    if report["ready"]:
        print("[Readiness] READY: strict offline prerequisites satisfied")
        return 0

    print("[Readiness] NOT READY: blockers detected")
    for item in report["blockers"]:
        print(f"[Readiness]   - {item}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
