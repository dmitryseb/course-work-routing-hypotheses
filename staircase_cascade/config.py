"""OpenRouter; оценка задержки через mlx_device_profile (пул совпадает с carrot-like/data_collection_sprout.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlx_device_profile.constants import est_device_seconds  # noqa: E402

TIER_ORDER: List[str] = ["llama1b", "llama3b", "qwen7b"]

OPENROUTER_MODEL_IDS = {
    "llama1b": "meta-llama/llama-3.2-1b-instruct",
    "llama3b": "meta-llama/llama-3.2-3b-instruct",
    "qwen7b": "qwen/qwen-2.5-7b-instruct",
}

BASE_URL = "https://openrouter.ai/api/v1"

JUDGE_MODEL = os.getenv(
    "OPENROUTER_JUDGE_MODEL",
    "meta-llama/llama-3.1-70b-instruct",
)

DEFAULT_MAX_TOKENS = 1536
DEFAULT_TEMPERATURE = 0.0
