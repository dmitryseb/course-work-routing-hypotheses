from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from openai import AsyncOpenAI

from config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    OPENROUTER_MODEL_IDS,
    TIER_ORDER,
    est_device_seconds,
)


def split_answer_and_confidence(raw: str) -> tuple[str, bool]:
    """Last non-empty line: CONFIDENCE HIGH/LOW; body above. On failure: full text, not confident."""
    text = (raw or "").strip()
    if not text:
        return "", False

    lines = text.splitlines()
    last_idx = len(lines) - 1
    while last_idx >= 0 and not lines[last_idx].strip():
        last_idx -= 1
    if last_idx < 0:
        return text, False

    last = lines[last_idx].strip()
    upper = last.upper()
    if upper == "CONFIDENCE: HIGH":
        return "\n".join(lines[:last_idx]).strip(), True
    if upper == "CONFIDENCE: LOW":
        return "\n".join(lines[:last_idx]).strip(), False
    if "CONFIDENCE:" in upper:
        val = upper.split(":", 1)[-1].strip()
        conf = val.startswith("HIGH")
        return "\n".join(lines[:last_idx]).strip(), conf

    return text, False


CASCADE_SYSTEM = """You are a helpful assistant. Answer the user's question fully and clearly.

At the very end, after your answer, add a new line containing EXACTLY ONE of these two lines (nothing else on that line):
CONFIDENCE: HIGH
CONFIDENCE: LOW

Use CONFIDENCE: HIGH only if you are fully sure your answer is factually correct and adequate for the task.
Use CONFIDENCE: LOW if you are uncertain, might be wrong, or the task is ambiguous."""


@dataclass
class StaircaseResult:
    final_answer: str
    total_device_seconds_est: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "total_device_seconds_est": round(self.total_device_seconds_est, 6),
        }


async def run_staircase(client: AsyncOpenAI, user_prompt: str) -> StaircaseResult:
    total_dev = 0.0
    final_answer = ""

    for idx, tier in enumerate(TIER_ORDER):
        model_id = OPENROUTER_MODEL_IDS[tier]
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": CASCADE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        usage = resp.usage
        comp_tok = int(getattr(usage, "completion_tokens", None) or 0) if usage else 0

        raw = resp.choices[0].message.content or ""
        answer, confident = split_answer_and_confidence(raw)

        total_dev += est_device_seconds(comp_tok, tier)

        if not answer.strip() and raw.strip():
            answer = raw.strip()
            confident = False

        is_last = idx == len(TIER_ORDER) - 1
        if confident:
            final_answer = answer.strip()
            if not final_answer and raw.strip():
                ls = raw.strip().splitlines()
                if ls and "CONFIDENCE:" in ls[-1].upper():
                    final_answer = "\n".join(ls[:-1]).strip()
                else:
                    final_answer = raw.strip()
            break
        if is_last:
            final_answer = answer.strip() or raw.strip()
            break

    return StaircaseResult(
        final_answer=final_answer,
        total_device_seconds_est=total_dev,
    )
