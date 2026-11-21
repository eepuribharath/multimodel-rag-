from app.config import Settings, get_settings
from app.generation.gate import GatePlan, QueryGate, looks_social
from app.generation.generate import (
    ABSTAIN_TEXT,
    GeminiGenerator,
    build_context,
    chitchat_answer,
    locator,
    resolve_answer,
)
from app.generation.local import LocalGenerator


def get_generator(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.generator == "local":
        return LocalGenerator(
            model_name=settings.local_gen_model,
            temperature=settings.gen_temperature,
            max_output_tokens=settings.gen_max_tokens,
        )
    return GeminiGenerator(
        model=settings.gen_model,
        temperature=settings.gen_temperature,
        max_output_tokens=settings.gen_max_tokens,
        api_key=settings.gemini_api_key,
    )


def get_gate(settings: Settings | None = None) -> QueryGate:
    settings = settings or get_settings()
    return QueryGate(model=settings.gen_model, api_key=settings.gemini_api_key or None)


__all__ = [
    "ABSTAIN_TEXT",
    "GatePlan",
    "GeminiGenerator",
    "LocalGenerator",
    "QueryGate",
    "build_context",
    "chitchat_answer",
    "get_gate",
    "get_generator",
    "locator",
    "looks_social",
    "resolve_answer",
]
