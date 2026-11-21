"""FastAPI entrypoint.

Run from backend/:  ../venv/bin/python -m uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import MEDIA_DIR, router


def create_app() -> FastAPI:
    app = FastAPI(title="multimodal-rag")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
    return app


app = create_app()
