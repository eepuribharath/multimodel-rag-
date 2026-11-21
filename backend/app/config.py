"""Central settings and pipeline constants.

Model IDs, the embedding dimension, and top-k values live here and nowhere
else. EMBED_DIM is permanent per index: changing it (or the model) requires a
full re-index, so treat a change here as a breaking migration.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Gemini Embedding 2 per-request limits, verified against
# ai.google.dev/gemini-api/docs/embeddings on 2026-07-02. Ingestion must
# produce chunks that fit; the embedding layer enforces them per request.
MAX_TEXT_TOKENS_PER_REQUEST = 8192
MAX_IMAGES_PER_REQUEST = 6
MAX_AUDIO_SECONDS = 180
MAX_VIDEO_SECONDS = 120
MAX_PDF_PAGES = 6


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Models ---
    gemini_api_key: str = ""
    embedder: str = "gemini"  # gemini | local | fake
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator: str = "gemini"  # gemini | local
    local_gen_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Public preview ID, verified 2026-07-02. When it GAs under a new name,
    # update here (and .env) only — nothing else may reference the ID.
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 3072  # PERMANENT per index. 1536/768 via Matryoshka truncation.
    embed_batch_size: int = 32  # text chunks per embed request
    # Stable GA tier (verified 2026-07-02); gemini-3.1-pro-preview is the
    # quality upgrade but has no API free tier
    gen_model: str = "gemini-3.5-flash"
    gen_temperature: float = 0.1
    gen_max_tokens: int = 1024
    history_turns: int = 4  # prior turns shown to the model per query

    # --- Reranker ---
    reranker: str = "local"  # local | fake (managed providers if eval picks one)
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # HF id, cached on first use
    cohere_api_key: str = ""
    rerank_top_k: int = 6

    # --- Retrieval ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "multimodal_rag"
    retrieve_top_k: int = 100
    rrf_k: int = 60

    # --- Observability ---
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
