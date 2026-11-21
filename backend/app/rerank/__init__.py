from app.config import Settings, get_settings
from app.rerank.base import Reranker
from app.rerank.fake import FakeReranker
from app.rerank.local import LocalReranker


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.reranker == "local":
        return LocalReranker(model_name=settings.rerank_model)
    if settings.reranker == "fake":
        return FakeReranker()
    raise ValueError(
        f"RERANKER={settings.reranker!r} is not implemented — 'local' and "
        "'fake' exist today; managed providers (cohere/voyage/zerank) get "
        "added if the spec 06 bake-off picks one"
    )


__all__ = ["Reranker", "FakeReranker", "LocalReranker", "get_reranker"]
