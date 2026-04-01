# Algorithm for Dynamic Selection of Artificial Intelligence Models

В данном репозитории находятся файлы для проверки гипотез для динамического выбора моделей на edge-устройствах.

## Папки

| Папка | Содержимое |
|--------|------------|
| **`carrot-like/`** | Скрипты для сбора данных из датасета Sprout (промпты, сбор ответов, судья), `judge_common.py`, `prompt_features.py`, ноутбук `experiments_quality_tokens.ipynb`. |
| **`staircase_cascade/`** | Каскад 1B→3B→7B через OpenRouter: `cascade.py`, `run_eval.py`, `judge_client.py`, ноутбук `hi_eval_analysis.ipynb`. |
| **`mlx_device_profile/`** | Локальные замеры MLX и константы скорости (`measure_mlx_metrics.py`, `constants.py`) для оценки времени на устройстве. |

Для запуска скриптов, которые вызывают модели через облако в OpenRouter, нужен API-ключ: задается в переменной окружения `OPENROUTER_API_KEY`.
