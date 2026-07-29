from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger("alma.llm_utils")


# Lazy loading clients and config to avoid circular import issues
_primary_client: OpenAI | None = None
_fallback_client: OpenAI | None = None
_resolved = False

def _resolve_failover_clients():
    global _primary_client, _fallback_client, _resolved
    if _resolved:
        return
    _resolved = True
    try:
        from src.config import (
            GEMINI_API_KEY,
            GEMINI_MODEL,
            GEMINI_OPENAI_BASE_URL,
            OPENAI_API_KEY,
            OPENAI_MODEL,
            OPENAI_BASE_URL,
        )

        # Primary: Gemini
        if GEMINI_API_KEY:
            # Clean up base url defaults if empty
            base_url = GEMINI_OPENAI_BASE_URL or None
            _primary_client = OpenAI(api_key=GEMINI_API_KEY, base_url=base_url)
            logger.info("Configured primary client as Gemini with model %s", GEMINI_MODEL)
        else:
            logger.warning("Gemini API key is not set; will fall back directly to OpenAI.")

        # Fallback: OpenAI
        if OPENAI_API_KEY:
            base_url = OPENAI_BASE_URL or None
            _fallback_client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)
            logger.info("Configured fallback client as OpenAI with model %s", OPENAI_MODEL)
    except Exception as exc:
        logger.error("Failed to resolve failover clients: %s", exc)


def chat_completion_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 5.0,
) -> Any:
    """Run a chat completion using Gemini as primary, falling back to OpenAI on failure."""
    _resolve_failover_clients()

    from src.config import GEMINI_MODEL, OPENAI_MODEL

    # Select client and model depending on whether primary is available
    use_primary = (_primary_client is not None)
    active_client = _primary_client if use_primary else _fallback_client
    active_model = GEMINI_MODEL if use_primary else OPENAI_MODEL

    if active_client is None:
        # Fall back to whatever client was passed in if config-based lookup failed
        active_client = client
        active_model = model

    delay = initial_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = active_client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=temperature,
            )
            if attempt > 1:
                logger.info(
                    "LLM request succeeded on retry %d/%d (%s)",
                    attempt,
                    max_retries,
                    active_model,
                )
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                # If we were using the primary client (Gemini) and it failed, trigger failover
                if use_primary and _fallback_client is not None:
                    logger.error(
                        "Primary LLM (%s) failed after %d attempts: %s. Switching to fallback (%s)...",
                        active_model,
                        attempt,
                        exc,
                        OPENAI_MODEL,
                    )
                    # Use fallback client with the same retry logic (single-attempt or multi-attempt)
                    try:
                        return _fallback_client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=messages,
                            temperature=temperature,
                        )
                    except Exception as fallback_exc:
                        logger.error(
                            "Fallback LLM (%s) also failed: %s",
                            OPENAI_MODEL,
                            fallback_exc,
                        )
                        raise fallback_exc from exc
                else:
                    logger.error(
                        "LLM request failed after %d attempts: %s", attempt, exc
                    )
                raise
            logger.warning(
                "LLM request attempt %d/%d failed (%s): %s; retrying in %.1fs",
                attempt,
                max_retries,
                active_model,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
    assert last_exc is not None
    raise last_exc
