"""Общий промпт и разбор ответа судьи (carrot-like + staircase_cascade). Менять здесь — пересобирать данные."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict

JUDGE_PROMPT = """I want you to act as an impartial judge for the correctness of a model's answer.
You will be provided with a Task, a Golden Answer (ground truth), and a Model's Answer.

YOUR GOAL:
Evaluate if the Model's Answer contains the correct information based on the Golden Answer.
IMPORTANT: Focus on SEMANTIC CORRECTNESS, not exact string matching.
- If the Model's Answer states the same fact as the Golden Answer but uses different wording, it is CORRECT.
- If the Model's Answer includes the Golden Answer within a longer sentence, it is CORRECT.
- Do not penalize for politeness, extra context, or formatting differences (e.g., "The answer is Hawaii" matches "Hawaii").
- Only give a low score if the core fact is wrong, missing, or contradicts the Golden Answer.

SCORING:
Pick a single correctness_score that MUST be one of: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (multiples of 0.1 only). Use the value that best fits the answer; do not restrict yourself to a few round numbers — intermediate tenths are encouraged when they better reflect partial correctness.

Reference anchors (examples, not an exclusive list):
1.0 = Perfect semantic match (correct fact, any wording).
0.8–0.9 = Very strong match with tiny gaps or extra harmless detail.
0.7 = Mostly correct with minor omissions or imprecise wording.
0.5–0.6 = Partially correct (some key truth, meaningful gaps).
0.3–0.4 = Mostly wrong but some relevant material.
0.1–0.2 = Almost entirely wrong, trace of relevance at most.
0.0 = Completely wrong, hallucinated, or irrelevant.

OUTPUT FORMAT:
You must output ONLY a valid JSON object with no markdown, no code blocks, no explanations outside the JSON.
Format:
{
  "correctness_score": <one of 0.0, 0.1, …, 1.0>,
  "justification": "<one brief sentence explaining the score>"
}
"""


def format_judge_user(prompt: str, golden: str, model_answer: str) -> str:
    return f"Task: {prompt}\n\nGolden Answer: {golden}\n\nModel's Answer: {model_answer}"


def parse_judge_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```\s*$", "", text).strip()
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not match:
        return {"quality_score": 0.0, "justification": "Failed to parse judge JSON"}
    try:
        data = json.loads(match.group(0))
        raw = data.get("correctness_score", data.get("quality_score", 0.0))
        score = float(raw)
        score = max(0.0, min(1.0, score))
        score = math.floor(score * 10.0 + 0.5) / 10.0
        score = max(0.0, min(1.0, score))
        return {
            "quality_score": score,
            "justification": str(data.get("justification", "")),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"quality_score": 0.0, "justification": "Invalid judge JSON"}
