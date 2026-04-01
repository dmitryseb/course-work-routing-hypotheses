import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from openai import APIError, AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
OR_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODELS = {
    "llama1b": "meta-llama/llama-3.2-1b-instruct",
    "llama3b": "meta-llama/llama-3.2-3b-instruct",
    "qwen7b": "qwen/qwen-2.5-7b-instruct",
}

OUTPUT_FILE = ROOT / "data" / "sprout_with_metrics.json"
CHECKPOINT_FILE = ROOT / "data" / "metrics_checkpoint.json"
PROMPTS_FILE = ROOT / "data" / "sprout_prompts.json"
CHECKPOINT_EVERY = 50

client = AsyncOpenAI(api_key=OR_API_KEY, base_url="https://openrouter.ai/api/v1")
embedder = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")


def encode_query_text(text: str) -> list[float]:
    v = embedder.encode(
        text,
        prompt_name="query",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return v.tolist()


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "results": []}


def save_checkpoint(results: list, processed: int) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": processed, "results": results}, f, ensure_ascii=False, indent=2)


with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    samples = json.load(f)


async def process_prompt(item, semaphore, retry=3):
    async with semaphore:
        prompt = item["prompt"]
        emb = encode_query_text(prompt)

        model_results = {}
        for name, model_id in MODELS.items():
            for attempt in range(retry):
                start = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1536,
                        temperature=0.0,
                    )
                    latency = time.perf_counter() - start
                    response_text = resp.choices[0].message.content
                    usage = resp.usage or SimpleNamespace(prompt_tokens=0, completion_tokens=0)

                    model_results[name] = {
                        "response": response_text,
                        "latency": round(latency, 3),
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                    }
                    break
                except APIError as e:
                    if attempt == retry - 1:
                        model_results[name] = {"error": str(e)}
                        print(f"{name} failed after retries: {e}")
                    else:
                        await asyncio.sleep(2**attempt)
        return {
            "prompt": prompt,
            "golden_answer": item["golden_answer"],
            "source_dataset": item["source_dataset"],
            "embedding": emb,
            "models": model_results,
        }


async def main():
    sem = asyncio.Semaphore(10)
    checkpoint = load_checkpoint()
    start_idx = int(checkpoint.get("processed", 0))
    results = list(checkpoint.get("results", []))

    if len(results) != start_idx:
        start_idx = len(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    with tqdm(total=n - start_idx, desc="Прогон 1536 токенов", unit="prompt") as pbar:
        for batch_start in range(start_idx, n, CHECKPOINT_EVERY):
            batch_end = min(batch_start + CHECKPOINT_EVERY, n)
            tasks = [process_prompt(samples[i], sem) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            save_checkpoint(results, batch_end)
            pbar.update(batch_end - batch_start)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


asyncio.run(main())
