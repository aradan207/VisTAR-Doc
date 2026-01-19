from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

from app.backend.api.agent import router as agent_router


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
    yologen_images = project_root / "yolo-gen" / "data" / "processed" / "images"
    if yologen_images.exists():
        app.mount("/api/media/yologen", StaticFiles(directory=str(yologen_images)), name="yologen_media")
        print(f"[Main] Mounted yologen images from: {yologen_images}")
    else:
        print(f"[Main] WARNING: yologen images directory not found: {yologen_images}")

    app.include_router(agent_router, prefix="/api/agent")
    
    @app.on_event("startup")
    async def startup_event():
        """Preload image search data on startup."""
        try:
            from app.backend.api.tools.image_search import _load_all_data
            print("[Main] Preloading image search data...")
            _load_all_data(force_reload=True)
            print("[Main] Image search data loaded successfully")
        except Exception as e:
            print(f"[Main] Failed to preload image search data: {e}")

    @app.get("/")
    async def root():
        return {"message": "Deep Research Agent API is running 🚀"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
