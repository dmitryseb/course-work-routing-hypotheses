import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("CARROT-LLM-Routing/SPROUT", split="train")
RNG = np.random.default_rng(42)
SAMPLES_PER_DATASET = 120


def _raw_dataset_key(ex) -> str:
    v = ex.get("dataset")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "unknown"


def _indices_by_dataset() -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for j in range(len(ds)):
        out[_raw_dataset_key(ds[j])].append(j)
    return dict(out)


def balanced_by_raw_dataset(per_theme: int, rng: np.random.Generator) -> list[int]:
    groups = _indices_by_dataset()
    themes = sorted(groups.keys())
    picked: set[int] = set()
    for key in themes:
        arr = np.array(groups[key], dtype=np.int64)
        if len(arr) == 0:
            continue
        k = min(per_theme, len(arr))
        take = rng.choice(arr, size=k, replace=False)
        picked.update(int(x) for x in take)
    return sorted(picked)


indices = balanced_by_raw_dataset(SAMPLES_PER_DATASET, RNG)
subset = ds.select(indices)

samples = []
for ex in subset:
    samples.append(
        {
            "prompt": ex["prompt"],
            "golden_answer": ex.get("golden_answer", ""),
            "source_dataset": ex.get("dataset", "unknown"),
        }
    )

with open(DATA_DIR / "sprout_prompts.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
