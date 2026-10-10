"""Multilingual knowledge editing dataset."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MultilingualFact:
    """Represents a fact with parallel queries in multiple languages."""
    english_prompt: str
    target_new: str
    ground_truth: str
    subject: str
    parallel_prompts: Dict[str, str]  # lang_code -> prompt


MULTILINGUAL_BENCHMARK_SAMPLES = [
    MultilingualFact(
        english_prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
        parallel_prompts={
            "es": "La Torre Eiffel esta en",
            "fr": "La Tour Eiffel est a",
            "de": "Der Eiffelturm befindet sich in",
            "it": "La Torre Eiffel si trova a",
        }
    ),
    MultilingualFact(
        english_prompt="Messi plays for",
        target_new="Miami",
        ground_truth="PSG",
        subject="Messi",
        parallel_prompts={
            "es": "Messi juega para",
            "fr": "Messi joue pour",
            "de": "Messi spielt fur",
            "it": "Messi gioca per",
        }
    ),
]


def load_multilingual_benchmark() -> List[MultilingualFact]:
    return list(MULTILINGUAL_BENCHMARK_SAMPLES)
