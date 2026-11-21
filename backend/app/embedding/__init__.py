from app.config import Settings, get_settings
from app.embedding.base import Embedder
from app.embedding.fake import FakeEmbedder
from app.embedding.gemini import GeminiEmbedder
from app.embedding.local import LocalEmbedder


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedder == "local":
        return LocalEmbedder(
            model_name=settings.local_embed_model,
            dim=settings.embed_dim,
        )
    if settings.gemini_api_key and settings.gemini_api_key != "paste-your-key-here":
        return GeminiEmbedder(
            model=settings.embed_model,
            dim=settings.embed_dim,
            api_key=settings.gemini_api_key,
            batch_size=settings.embed_batch_size,
        )
    return FakeEmbedder(dim=settings.embed_dim)


__all__ = ["Embedder", "FakeEmbedder", "GeminiEmbedder", "LocalEmbedder", "get_embedder"]
