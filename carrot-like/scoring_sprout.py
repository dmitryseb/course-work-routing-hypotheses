import asyncio
import json
import os

from openai import APIError, AsyncOpenAI
from tqdm.asyncio import tqdm

from judge_common import JUDGE_PROMPT, format_judge_user, parse_judge_json

BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "meta-llama/llama-3.1-70b-instruct")

INPUT_FILE = "data/sprout_with_metrics.json"
OUTPUT_FILE = "data/sprout_with_scores_judge.json"
CHECKPOINT_FILE = "data/rescore_checkpoint.json"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MAX_CONCURRENT = 8

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


async def judge_response(prompt, golden, model_answer, semaphore):
    async with semaphore:
        if not str(golden).strip() or not str(model_answer).strip():
            return {"quality_score": 0.0, "justification": "Empty answer"}

        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": format_judge_user(prompt, golden, model_answer)},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            return parse_judge_json(text)
        except APIError as e:
            return {"quality_score": 0.0, "justification": f"APIError: {e}"}


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "data": []}


def save_checkpoint(data, processed_count):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": processed_count, "data": data}, f, ensure_ascii=False, indent=2)


async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoint = load_checkpoint()
    processed_count = checkpoint.get("processed", 0)
    results = checkpoint.get("data", [])

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    for i in tqdm(range(processed_count, len(data)), desc="Оценка судьёй"):
        item = data[i]
        prompt = item["prompt"]
        golden = item.get("golden_answer", "")

        for _, model_data in item["models"].items():
            if "error" in model_data:
                continue

            response = model_data.get("response", "")
            judge_result = await judge_response(prompt, golden, response, semaphore)

            model_data["quality_score"] = judge_result["quality_score"]
            model_data["justification"] = judge_result["justification"]
            model_data["judge_used"] = JUDGE_MODEL

        results.append(item)

        if (i + 1) % 50 == 0:
            save_checkpoint(results, i + 1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
