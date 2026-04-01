"""Локальный MLX: TTFT (константа на модель), decode (токены вывода/с), оценка RAM.

Оценка времени одного ответа: T ≈ MODEL_TTFT_SEC[tier] + completion_tokens / MODEL_SPEEDS[tier].

Перекалибровать decode: `python -m mlx_device_profile.measure_mlx_metrics` (средний generation t/s).
TTFT — задать вручную по замерам или взять среднее из вывода скрипта (как время до первого токена mlx).
"""

from __future__ import annotations

MODEL_TTFT_SEC = {"llama1b": 0.09, "llama3b": 0.23, "qwen7b": 0.52}
MODEL_SPEEDS = {"llama1b": 82.0, "llama3b": 47.0, "qwen7b": 22.0}
MODEL_MEMORY_MB = {"llama1b": 1200, "llama3b": 2400, "qwen7b": 4800}


def est_device_seconds(completion_tokens: int, tier: str) -> float:
    """Оценка времени одного forward+decode на устройстве (сек)."""
    ttft = float(MODEL_TTFT_SEC.get(tier, 0.0))
    dpt = float(MODEL_SPEEDS.get(tier, 1.0))
    dec = (float(completion_tokens) / dpt) if dpt > 0 else 0.0
    return ttft + dec
