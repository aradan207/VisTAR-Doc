from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

from app.backend.api.agent import router as agent_router
from app.backend.core.runtime_paths import (
    offline_mode_enabled,
    require_image_retrieval,
    require_semantic_model_cache,
    required_image_artifact_paths,
    resolve_yologen_processed_candidates,
    resolve_yologen_processed_dir,
    sentence_transformers_cache_available,
)


def _offline_mode_enabled() -> bool:
    return offline_mode_enabled()


def _strict_offline_image_mode() -> bool:
    return _offline_mode_enabled() and require_image_retrieval()


def _find_yologen_processed_dir(project_root: Path) -> Path | None:
    """Find processed yologen data directory from env overrides and fallbacks."""
    return resolve_yologen_processed_dir(project_root)


def _find_yologen_images_dir(processed_dir: Path | None) -> Path | None:
    """Return images directory under processed yologen root if present."""
    if processed_dir is None:
        return None
    images_dir = processed_dir / "images"
    return images_dir if images_dir.exists() else None


def _log_yologen_resolution(project_root: Path, processed_dir: Path | None) -> None:
    print("[Main] Yologen path resolution candidates:")
    for candidate in resolve_yologen_processed_candidates(project_root):
        print(f"[Main]   - {candidate} exists={candidate.exists()}")
    if processed_dir is None:
        print("[Main] Yologen processed root: not found")
    else:
        print(f"[Main] Yologen processed root: {processed_dir}")


def _validate_startup_prerequisites(project_root: Path, processed_dir: Path | None) -> None:
    """Validate strict offline prerequisites before serving requests.

    In strict offline image mode, missing artifacts/caches are treated as fatal.
    """
    strict_mode = _strict_offline_image_mode()

    missing_artifacts = []
    if processed_dir is None:
        missing_artifacts.append("yologen processed directory")
    else:
        for path in required_image_artifact_paths(processed_dir):
            if not path.exists():
                missing_artifacts.append(str(path))

    if missing_artifacts:
        print("[Main] WARNING: image retrieval artifacts missing:")
        for item in missing_artifacts:
            print(f"[Main]   - {item}")

    # Validate local HF caches to avoid accidental network requests offline.
    semantic_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    missing_model_cache = []
    if _offline_mode_enabled() and require_semantic_model_cache():
        for model_name in semantic_models:
            if not sentence_transformers_cache_available(model_name):
                missing_model_cache.append(model_name)

    if missing_model_cache:
        print("[Main] WARNING: local sentence-transformers caches missing:")
        for model_name in missing_model_cache:
            print(f"[Main]   - {model_name}")
        print("[Main]   Populate HF_HOME/SENTENCE_TRANSFORMERS_HOME on a connected machine, then transfer offline.")

    if strict_mode and (missing_artifacts or missing_model_cache):
        reasons = []
        if missing_artifacts:
            reasons.append("image artifacts are missing")
        if missing_model_cache:
            reasons.append("semantic model cache is missing")
        raise RuntimeError(
            "Strict offline image mode blocked startup because "
            + " and ".join(reasons)
            + ". Set REQUIRE_IMAGE_RETRIEVAL=false to allow degraded startup."
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deep Research Agent API",
        description="Backend API to run the autonomous reasoning agent",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for images
    project_root = Path(__file__).parent.parent.parent.parent
    yologen_processed = _find_yologen_processed_dir(project_root)
    _log_yologen_resolution(project_root, yologen_processed)

    yologen_images = _find_yologen_images_dir(yologen_processed)
    if yologen_images is not None:
        app.mount("/api/media/yologen", StaticFiles(directory=str(yologen_images)), name="yologen_media")
        print(f"[Main] Mounted yologen images from: {yologen_images}")
    else:
        print("[Main] WARNING: yologen images directory not found under processed data root")

    app.include_router(agent_router, prefix="/api/agent")
    
    @app.on_event("startup")
    async def startup_event():
        """Preload image search data and emit offline readiness diagnostics."""
        print(f"[Main] OFFLINE_MODE={'enabled' if _offline_mode_enabled() else 'disabled'}")
        print(f"[Main] REQUIRE_IMAGE_RETRIEVAL={'enabled' if require_image_retrieval() else 'disabled'}")
        print(f"[Main] REQUIRE_SEMANTIC_MODEL_CACHE={'enabled' if require_semantic_model_cache() else 'disabled'}")

        _validate_startup_prerequisites(project_root, yologen_processed)

        try:
            from app.backend.api.tools.image_search import _load_all_data
            print("[Main] Preloading image search data...")
            _load_all_data(force_reload=True)
            print("[Main] Image search data loaded successfully")
        except Exception as e:
            print(f"[Main] Failed to preload image search data: {e}")
            if _strict_offline_image_mode():
                raise

    @app.get("/")
    async def root():
        return {"message": "Deep Research Agent API is running 🚀"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
