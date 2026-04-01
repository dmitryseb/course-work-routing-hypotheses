"""Скалярные признаки промпта для carrot-like / ноутбуков (20 чисел: структура и пунктуация)."""
from __future__ import annotations

import math
import re
from collections import Counter


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def avg_sentence_length(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def prompt_features(text: str) -> list[float]:
    text_lower = text.lower()
    words = text_lower.split()
    n_words = len(words) if words else 1
    n_chars = len(text)
    features: list[float] = [
        float(n_chars),
        float(n_words),
        float(text.count("\n")),
        n_chars / (n_words + 1),
        sum(c.isdigit() for c in text) / (n_chars + 1),
        float(len(re.findall(r"[\+\-\*/=^]", text))),
        float(len(re.findall(r"\d+", text))),
        float(bool(re.search(r"\d\s*[\+\-\*/=]", text))),
        sum(1 for w in words if len(w) > 8) / (n_words + 1),
        sum(len(w) for w in words) / (n_words + 1),
        len(set(words)) / (n_words + 1),
        len(set(text_lower)) / (n_chars + 1),
        text.count(" ") / (n_words + 1),
        char_entropy(text),
        avg_sentence_length(text),
    ]
    features += [
        float(text.count(".")),
        float(text.count("?")),
        float(text.count(",")),
        float(text.count(";") + text.count(":") + text.count("(")),
        len(text_lower.split(" ")) / (text.count(".") + 1),
    ]
    return features
