"""Shared retry posture for Gemini API calls.

429 (quota — preview tiers are tight) and 5xx retry with exponential
backoff; any other error fails fast.
"""

from google.genai import errors
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in _RETRYABLE_CODES


def build_retryer(
    max_attempts: int = 6, backoff_base: float = 0.5, backoff_max: float = 30.0
) -> Retrying:
    return Retrying(
        retry=retry_if_exception(is_retryable_api_error),
        wait=wait_exponential(multiplier=backoff_base, max=backoff_max),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
