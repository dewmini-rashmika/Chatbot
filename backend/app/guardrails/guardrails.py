"""
Input & Output Guardrails.

These are fast, deterministic checks that run BEFORE and AFTER the LLM,
providing a safety layer that doesn't depend on the model's behavior.

Why deterministic guardrails + LLM reviewer?
- Deterministic rules are cheap and instant (regex/keyword checks).
- LLM-based reviewer (reviewer_agent) handles nuanced hallucination/quality checks.
- Two-layer defence: fast regex blocks obvious violations before they reach the model.
"""

import re
from dataclasses import dataclass

# ── Compiled patterns ─────────────────────────────────────────────────────────

_JAILBREAK = re.compile(
    r"\b(ignore (all |previous |above )?instructions?|you are now|pretend you are"
    r"|act as (a |an )?|disregard your|forget your rules|system prompt"
    r"|reveal your (system |base )?prompt)\b",
    re.IGNORECASE,
)

_LYRIC_TRIGGERS = re.compile(
    r"\b(full lyrics?|all the lyrics?|complete lyrics?|write me the lyrics?"
    r"|reproduce the lyrics?|copy the lyrics?|print the lyrics?)\b",
    re.IGNORECASE,
)

_OFF_TOPIC = re.compile(
    r"\b(write my code|debug my|sql injection|hack|exploit|malware"
    r"|generate password|credit card|ssn|social security"
    r"|medical advice|legal advice|invest in stock)\b",
    re.IGNORECASE,
)

# Output: catch suspiciously long quoted strings (possible verbatim lyrics)
_LYRIC_LINE_PATTERN = re.compile(r'(["\'])([^"\']{200,})\1')


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""


def check_input_guardrails(user_message: str) -> GuardrailResult:
    """
    Run all input guardrails on the user's message.
    Returns immediately on the first violation found.
    """
    if _JAILBREAK.search(user_message):
        return GuardrailResult(
            passed=False,
            reason="Prompt injection or jailbreak attempt detected.",
        )

    if _LYRIC_TRIGGERS.search(user_message):
        return GuardrailResult(
            passed=False,
            reason=(
                "I cannot reproduce full song lyrics due to copyright restrictions. "
                "I can discuss lyrical themes, structure, and style instead."
            ),
        )

    if _OFF_TOPIC.search(user_message):
        return GuardrailResult(
            passed=False,
            reason="I'm specialized in music knowledge and discovery. I can't help with that topic.",
        )

    return GuardrailResult(passed=True)


def check_output_guardrails(response_text: str) -> GuardrailResult:
    """
    Run output guardrails on the generated response.
    The LLM-based Reviewer Agent handles advanced copyright checks (lyrics),
    so this deterministic gate is kept minimal to prevent false positives.
    """
    if _LYRIC_LINE_PATTERN.search(response_text):
        return GuardrailResult(
            passed=False,
            reason=(
                "I cannot reproduce full song lyrics due to copyright restrictions. "
                "I can discuss lyrical themes, structure, and style instead."
            ),
        )

    return GuardrailResult(passed=True)
