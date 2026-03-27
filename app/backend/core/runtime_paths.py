from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value not in {"0", "false", "no", "off"}


def offline_mode_enabled() -> bool:
    return env_flag("OFFLINE_MODE", True)


def require_image_retrieval() -> bool:
    # Strict by default in offline mode because image retrieval is core behavior.
    return env_flag("REQUIRE_IMAGE_RETRIEVAL", offline_mode_enabled())


def require_semantic_model_cache() -> bool:
    # Enforce deterministic no-network behavior for sentence-transformer models.
    return env_flag("REQUIRE_SEMANTIC_MODEL_CACHE", offline_mode_enabled())


def resolve_yologen_processed_candidates(project_root: Path) -> List[Path]:
    candidates: List[Path] = []

    yologen_data_root = os.getenv("YOLOGEN_DATA_ROOT", "").strip()
    if yologen_data_root:
        candidates.append(Path(yologen_data_root).expanduser())

    yologen_root = os.getenv("YOLOGEN_ROOT", "").strip()
    if yologen_root:
        base = Path(yologen_root).expanduser()
        if base.name.lower() == "processed":
            candidates.append(base)
        else:
            candidates.append(base / "data" / "processed")

    candidates.extend(
        [
            project_root.parent / "vlm-yolo-detector" / "data" / "processed",
            project_root / "vlm-yolo-detector" / "data" / "processed",
        ]
    )
    return candidates


def resolve_yologen_processed_dir(project_root: Path) -> Optional[Path]:
    for candidate in resolve_yologen_processed_candidates(project_root):
        if candidate.exists():
            return candidate
    return None


def required_image_artifact_paths(processed_dir: Path) -> List[Path]:
    return [
        processed_dir / "image_index.json",
        processed_dir / "image_embeddings.npy",
        processed_dir / "embedding_mapping.json",
        processed_dir / "images",
    ]


def sentence_transformers_cache_roots() -> List[Path]:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_hf = repo_root / ".cache" / "huggingface"
    repo_st = repo_root / ".cache" / "torch" / "sentence_transformers"

    return [
        Path(os.getenv("HF_HOME", repo_hf)) / "hub",
        Path(os.getenv("SENTENCE_TRANSFORMERS_HOME", repo_st)),
        Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        Path(
            os.getenv(
                "SENTENCE_TRANSFORMERS_HOME",
                Path.home() / ".cache" / "torch" / "sentence_transformers",
            )
        ),
    ]


def sentence_transformers_cache_available(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    leaf = normalized.split("/")[-1]
    tokens = {leaf, normalized.replace("/", "-"), normalized.replace("/", "--")}

    for root in sentence_transformers_cache_roots():
        if not root.exists():
            continue

        if root.name == "hub":
            for entry in root.glob("models--*"):
                name = entry.name.lower()
                if any(token in name for token in tokens):
                    return True
        else:
            for entry in root.glob("**/*"):
                if entry.is_dir() and any(token in entry.name.lower() for token in tokens):
                    return True
    return False
