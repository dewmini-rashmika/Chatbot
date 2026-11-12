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

import os
import litellm
from litellm import acompletion
from app.core.config import settings

# ── Provider key injection ────────────────────────────────────────────────────
# LiteLLM reads these from the environment automatically, but we also set them
# explicitly here so the calling code never has to think about it.

def _configure_litellm() -> None:
    """Push all API keys into the environment for LiteLLM to pick up."""
    key_map = {
        "GROQ_API_KEY":      settings.groq_api_key,
        "GOOGLE_API_KEY":    settings.google_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "OPENAI_API_KEY":    settings.openai_api_key,
        "DEEPSEEK_API_KEY":  settings.deepseek_api_key,
    }
    for env_var, value in key_map.items():
        if value:
            os.environ[env_var] = value

    # Suppress verbose LiteLLM success logs in production
    litellm.suppress_debug_info = settings.app_env == "production"
    litellm.set_verbose = settings.app_env == "development"


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
    **kwargs,
):
    """
    Async LiteLLM completion call routed by agent role.

    Parameters
    ----------
    role       : One of the keys in ROLE_MODEL_MAP (e.g. "supervisor", "rag").
    messages   : OpenAI-format message list.
    temperature: Sampling temperature (default 0.7).
    max_tokens : Optional token cap.
    **kwargs   : Forwarded verbatim to litellm.acompletion.

    Returns
    -------
    litellm.ModelResponse — access the text via
        response.choices[0].message.content
    """
    model = ROLE_MODEL_MAP.get(role)
    if not model:
        raise ValueError(
            f"Unknown LLM role '{role}'. "
            f"Valid roles: {list(ROLE_MODEL_MAP.keys())}"
        )

    call_kwargs: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    return await acompletion(**call_kwargs)


def get_model_for_role(role: str) -> str:
    """Return the model string for a given role (useful for logging/tracing)."""
    model = ROLE_MODEL_MAP.get(role)
    if not model:
        raise ValueError(f"Unknown LLM role '{role}'.")
    return model
