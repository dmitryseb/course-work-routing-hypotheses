from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from cascade import run_staircase
from config import BASE_URL, JUDGE_MODEL
from judge_client import judge_answer

_STAIRCASE = Path(__file__).resolve().parent
_DEFAULT_IN = _STAIRCASE.parent / "carrot-like" / "data" / "sprout_prompts.json"
_DEFAULT_OUT = _STAIRCASE / "output" / "staircase_eval.json"


async def process_one(client, item: dict, sem_cascade, sem_judge) -> dict:
    prompt = item.get("prompt", "")
    golden = item.get("golden_answer", "")
    async with sem_cascade:
        sc = await run_staircase(client, prompt)
    async with sem_judge:
        j = await judge_answer(client, JUDGE_MODEL, prompt, golden, sc.final_answer)
    return {
        "prompt": prompt,
        "golden_answer": golden,
        "staircase": sc.to_dict(),
        "judge": {
            "quality_score": j["quality_score"],
            "justification": j["justification"],
            "judge_model": j.get("judge_model", JUDGE_MODEL),
        },
    }


async def main_async(limit: int | None, seed: int | None) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY")
    inp = _DEFAULT_IN
    data = json.loads(inp.read_text(encoding="utf-8"))
    if limit is not None:
        if seed is not None:
            data = data.copy()
            random.Random(seed).shuffle(data)
        data = data[:limit]
    client = AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=BASE_URL)
    sem_c = asyncio.Semaphore(2)
    sem_j = asyncio.Semaphore(1)
    results = []
    for item in tqdm(data, desc="HI eval"):
        row = await process_one(client, item, sem_c, sem_j)
        results.append(row)
    outp = _DEFAULT_OUT
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Staircase cascade (1B→3B→7B) + LLM judge via OpenRouter.")
    p.add_argument("--limit", type=int, default=None, help="Max prompts (optional with --seed shuffle)")
    p.add_argument("--seed", type=int, default=None, help="Shuffle seed (only with --limit)")
    args = p.parse_args()
    asyncio.run(main_async(args.limit, args.seed))


if __name__ == "__main__":
    main()
