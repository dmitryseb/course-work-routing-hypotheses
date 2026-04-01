from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from openai import RateLimitError

_REPO = Path(__file__).resolve().parent.parent
_CARROT = _REPO / "carrot-like"
if str(_CARROT) not in sys.path:
    sys.path.insert(0, str(_CARROT))

from judge_common import JUDGE_PROMPT, format_judge_user, parse_judge_json

JUDGE_MAX_RETRIES = 8
JUDGE_BASE_DELAY_S = 2.0


async def judge_answer(client, model: str, prompt: str, golden: str, model_answer: str):
    if not str(golden).strip() or not str(model_answer).strip():
        return {"quality_score": 0.0, "justification": "Empty golden or answer"}

    last_err = None
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": format_judge_user(prompt, golden, model_answer),
                    },
                ],
                max_tokens=1536,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            out = parse_judge_json(text)
            out["judge_model"] = model
            return out
        except RateLimitError as e:
            last_err = e
            await asyncio.sleep(min(90.0, JUDGE_BASE_DELAY_S * (2**attempt)))
    return {
        "quality_score": 0.0,
        "justification": f"judge RateLimit after {JUDGE_MAX_RETRIES} retries: {last_err}",
        "judge_model": model,
    }
