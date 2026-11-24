"""
LiteLLM Gateway — centralised LLM client factory.

Why LiteLLM?
- One unified interface for 100+ providers (Groq, Anthropic, Google, OpenAI, DeepSeek…).
- Swap any model by changing a single env var — no code changes needed.
- Built-in retry, fallback, and cost tracking.

Usage
-----
    from app.core.llm_factory import call_llm

    response = await call_llm(
        role="supervisor",          # maps to settings.supervisor_model
        messages=[{"role": "user", "content": "Hello"}],
    )
    text = response.choices[0].message.content

All agent/service modules should use `call_llm(role=..., messages=...)` instead
of instantiating provider-specific clients directly.
"""

import asyncio
import logging
import os
import litellm
from litellm import acompletion
from litellm.exceptions import ServiceUnavailableError, RateLimitError, NotFoundError
from app.core.config import settings

log = logging.getLogger(__name__)

# ── Provider key injection ────────────────────────────────────────────────────
# LiteLLM reads these from the environment automatically, but we also set them
# explicitly here so the calling code never has to think about it.

def _configure_litellm() -> None:
    """Push all API keys into the environment for LiteLLM to pick up."""
    key_map = {
        "GROQ_API_KEY":      settings.groq_api_key,
        "GOOGLE_API_KEY":    settings.google_api_key,
        "GEMINI_API_KEY":    settings.google_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "OPENAI_API_KEY":    settings.openai_api_key,
        "DEEPSEEK_API_KEY":  settings.deepseek_api_key,
    }
    for env_var, value in key_map.items():
        if value:
            os.environ[env_var] = value

    # Drop unsupported params (e.g. temperature on Gemini) instead of erroring
    litellm.drop_params = True

    # Suppress verbose LiteLLM success logs in production
    litellm.suppress_debug_info = settings.app_env == "production"
    if settings.app_env == "development":
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        os.environ["LITELLM_LOG"] = "WARNING"

    # Global fallback: if the primary model returns a transient error (503/429),
    # LiteLLM automatically retries with the fallback model list.
    litellm.fallbacks = [{"groq/qwen/qwen3.8-27b": ["gemini/gemini-3.8-flash"]}]


_configure_litellm()

# ── Role → model string mapping ───────────────────────────────────────────────
# Every role resolves to a LiteLLM model string from settings.
# Format: "<provider>/<model-name>"  e.g. "groq/llama-3.3-70b-versatile"

ROLE_MODEL_MAP: dict[str, str] = {
    # 1. Supervisor  — ultra-low latency classification
    "supervisor":   settings.supervisor_model,
    # 2. Text-to-SQL — high SQL accuracy
    "sql":          settings.sql_model,
    # 3. RAG Generator — large context, natural synthesis
    "rag":          settings.rag_model,
    # 4. Reviewer / Judge — deep reasoning quality gate
    "reviewer":     settings.reviewer_model,
    # 5. Memory Extractor — simple JSON extraction, background task
    "memory":       settings.memory_model,
    # 6. Chat Agent — casual conversation
    "chat":         settings.chat_model,
    # 7. Creative Agent — songwriting / lyrics / poetry
    "creative":     settings.creative_model,
    # 8. Tool Agent — function calling / live API orchestration
    "tool":         settings.tool_model,
}


async def call_llm(
    role: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_retries: int = 3,
    **kwargs,
):
    """
    Async LiteLLM completion call routed by agent role.

    Tries models in order: primary → GROQ_FALLBACK → raises.

    Parameters
    ----------
    role        : One of the keys in ROLE_MODEL_MAP (e.g. "supervisor", "rag").
    messages    : OpenAI-format message list.
    temperature : Sampling temperature (default 0.7).
    max_tokens  : Optional token cap.
    max_retries : Retry attempts per model on 503 (default 3).
    **kwargs    : Forwarded verbatim to litellm.acompletion.
    """
    primary_model = ROLE_MODEL_MAP.get(role)
    if not primary_model:
        raise ValueError(
            f"Unknown LLM role '{role}'. "
            f"Valid roles: {list(ROLE_MODEL_MAP.keys())}"
        )

    # Groq free tier caps output at 1000 OTPM. Groq pre-checks max_tokens
    # against this limit before running — always set it explicitly for Groq.
    GROQ_FALLBACK   = "groq/qwen/qwen3.8-27b"
    GROQ_MAX_TOKENS = 900  # safely under 1000 OTPM single-request limit

    def _capped(model: str, mt: int | None) -> int | None:
        """Cap max_tokens to GROQ_MAX_TOKENS for any groq/ model."""
        if model.startswith("groq/"):
            return min(mt or GROQ_MAX_TOKENS, GROQ_MAX_TOKENS)
        return mt

    # Build the model attempt list: skip fallback if primary IS the fallback
    models_to_try: list[tuple[str, int | None]] = [
        (primary_model, _capped(primary_model, max_tokens)),
    ]
    if primary_model != GROQ_FALLBACK:
        models_to_try.append((GROQ_FALLBACK, _capped(GROQ_FALLBACK, max_tokens)))

    last_exc: Exception | None = None

    for model, model_max_tokens in models_to_try:
        call_kwargs: dict = {
            "model":    model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if model_max_tokens is not None:
            call_kwargs["max_tokens"] = model_max_tokens

        # Groq OTPM is a sliding 60s window shared across concurrent calls.
        # A short delay before each Groq call prevents bursts from exceeding 1000/min.
        if model.startswith("groq/"):
            await asyncio.sleep(2.0)

        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                return await acompletion(**call_kwargs)
            except (NotFoundError, RateLimitError) as exc:
                err_str = str(exc)
                # Groq sliding-window: "Used X, Requested Y" — wait and retry
                # Groq single-request too large: "exceed the enforced limit" — give up
                # Gemini daily quota: "exceeded your current quota" — give up immediately
                is_groq_window = (
                    model.startswith("groq/")
                    and "exceed the enforced limit" not in err_str
                    and attempt < max_retries
                )
                if is_groq_window:
                    wait = 60.0 / max_retries
                    log.warning(
                        f"call_llm.rate_window model={model} attempt={attempt} "
                        f"— retrying in {wait:.0f}s",
                    )
                    await asyncio.sleep(wait)
                    continue
                # Everything else: give up on this model, try the next
                log.warning(
                    f"call_llm.skip_model [{exc.__class__.__name__}] model={model}",
                    extra={"error": err_str[:160]},
                )
                last_exc = exc
                break
            except ServiceUnavailableError as exc:
                last_exc = exc
                if attempt == max_retries:
                    log.warning(
                        f"call_llm.exhausted_retries model={model} — moving on",
                    )
                    break  # try next model
                log.warning(
                    "call_llm.transient_error",
                    extra={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "delay_s": delay,
                        "model": model,
                        "error": str(exc)[:120],
                    },
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    raise last_exc  # all models exhausted


def get_model_for_role(role: str) -> str:
    """Return the model string for a given role (useful for logging/tracing)."""
    model = ROLE_MODEL_MAP.get(role)
    if not model:
        raise ValueError(f"Unknown LLM role '{role}'.")
    return model
