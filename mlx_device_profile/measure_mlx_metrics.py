"""Калибровка decode (токены вывода/с) и подсказка TTFT через mlx_lm.stream_generate.

На первом чанке mlx: prompt_tps = prompt_tokens / wall_time до первого токена.
Среднее (prompt_tokens / prompt_tps) по сэмплам ≈ типичный TTFT для калибровочных промптов — см. вывод MODEL_TTFT_SEC.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

_REPO = Path(__file__).resolve().parent.parent
PROMPTS_FILE = _REPO / "carrot-like" / "data" / "sprout_with_scores_judge.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "sprout_mlx_metrics.json"
CHECKPOINT_FILE = Path(__file__).resolve().parent / "mlx_checkpoint.json"
SAMPLE_SIZE = 100
MAX_TOKENS = 256

MODELS_CONFIG = {
    "llama1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "qwen7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
}


def estimate_model_memory(model_path):
    defaults = {
        "llama-3.2-1b": 1200,
        "llama-3.2-3b": 2400,
        "qwen2.5-7b": 4800,
    }
    model_path_lower = model_path.lower()
    for key, value in defaults.items():
        if key in model_path_lower:
            return value
    return 2000


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "results": []}


def save_checkpoint(data, processed_count):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": processed_count, "results": data}, f, ensure_ascii=False, indent=2)


def measure_model_mlx(prompt, model_name, model, tokenizer, model_path):
    input_token_count = 0

    def fail(msg: str) -> dict:
        print(f"error {model_name}: {msg}")
        return {
            "success": False,
            "error": msg,
            "prefill_sec": None,
            "prefill_tokens_per_sec": None,
            "total_latency": None,
            "tokens_generated": 0,
            "tokens_per_sec": 0.0,
            "gpu_memory_mb": 0.0,
            "input_tokens": input_token_count,
        }

    try:
        input_token_count = len(tokenizer.encode(prompt))
        mx.metal.clear_cache()
        time.sleep(0.1)
        mx.eval(model.parameters())

        t0 = time.perf_counter()
        prefill_sec = prefill_tps = None
        gen_tokens = 0

        for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
            if prefill_sec is None and getattr(resp, "prompt_tps", 0) and resp.prompt_tps > 0:
                prefill_tps = float(resp.prompt_tps)
                prefill_sec = float(resp.prompt_tokens) / prefill_tps
            gen_tokens = int(resp.generation_tokens)
            if resp.finish_reason:
                break

        wall = time.perf_counter() - t0
        ps = prefill_sec if prefill_sec is not None else 0.0
        pt = prefill_tps if prefill_tps is not None else 0.0
        decode_sec = max(wall - ps, 1e-9)

        return {
            "success": True,
            "error": None,
            "prefill_sec": round(ps, 4),
            "prefill_tokens_per_sec": round(pt, 2),
            "total_latency": round(wall, 4),
            "tokens_generated": gen_tokens,
            "tokens_per_sec": round(gen_tokens / decode_sec, 2),
            "gpu_memory_mb": estimate_model_memory(model_path),
            "input_tokens": input_token_count,
        }
    except Exception as e:
        return fail(str(e))


def main():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    samples = all_data[:SAMPLE_SIZE]

    checkpoint = load_checkpoint()
    processed_count = checkpoint.get("processed", 0)
    results = checkpoint.get("results", [])

    loaded_models = {}
    for name, path in MODELS_CONFIG.items():
        try:
            model, tokenizer = load(path)
            loaded_models[name] = {"model": model, "tokenizer": tokenizer, "path": path}
        except Exception as e:
            print(f"load failed {name}: {e}")

    n = len(samples)
    for i in range(processed_count, n):
        item = samples[i]
        prompt = item["prompt"]
        print(f"[{i + 1}/{n}]", flush=True)

        item_result = {
            "prompt_id": i,
            "prompt": prompt,
            "golden_answer": item.get("golden_answer", ""),
            "models": {},
        }

        for model_name, model_data in loaded_models.items():
            mx.metal.clear_cache()

            metrics = measure_model_mlx(
                prompt=prompt,
                model_name=model_name,
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                model_path=model_data["path"],
            )

            item_result["models"][model_name] = metrics

        results.append(item_result)

        if (i + 1) % 10 == 0:
            save_checkpoint(results, i + 1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if CHECKPOINT_FILE.exists():
        os.remove(CHECKPOINT_FILE)

    all_metrics = {
        m: {"prefill_sec": [], "prefill_tps": [], "latency": [], "speed": [], "memory": []}
        for m in MODELS_CONFIG.keys()
    }
    for item in results:
        for m_name, m_metrics in item["models"].items():
            if m_metrics["success"]:
                all_metrics[m_name]["prefill_sec"].append(m_metrics["prefill_sec"])
                pts = m_metrics.get("prefill_tokens_per_sec") or 0.0
                if pts > 0:
                    all_metrics[m_name]["prefill_tps"].append(float(pts))
                all_metrics[m_name]["latency"].append(m_metrics["total_latency"])
                all_metrics[m_name]["speed"].append(m_metrics["tokens_per_sec"])
                all_metrics[m_name]["memory"].append(m_metrics["gpu_memory_mb"])

    parts = [f"mlx {mx.__version__}", f"n={len(results)}", f"out={OUTPUT_FILE}"]
    config_ttft = []
    config_speed = []
    for m_name in MODELS_CONFIG.keys():
        am = all_metrics[m_name]
        if not am["speed"]:
            continue
        mean_prefill_s = sum(am["prefill_sec"]) / len(am["prefill_sec"])
        mean_prefill_tps = sum(am["prefill_tps"]) / len(am["prefill_tps"]) if am["prefill_tps"] else 0.0
        mean_speed = sum(am["speed"]) / len(am["speed"])
        config_ttft.append(f'"{m_name}": {mean_prefill_s:.4f}')
        parts.append(
            f"{m_name}: ttft≈{mean_prefill_s:.4f}s "
            f"({mean_prefill_tps:.1f} prompt tok/s mlx) "
            f"lat={sum(am['latency'])/len(am['latency']):.4f}s "
            f"{mean_speed:.1f} t/s (decode)"
        )
        config_speed.append(f'"{m_name}": {mean_speed:.2f}')
    print(" | ".join(parts))
    if config_ttft:
        print()
        print("Вставьте в mlx_device_profile/constants.py (TTFT — среднее время до 1-го токена на этих промптах):")
        print("MODEL_TTFT_SEC = {" + ", ".join(config_ttft) + "}")
        print("MODEL_SPEEDS = {" + ", ".join(config_speed) + "}")


if __name__ == "__main__":
    main()
