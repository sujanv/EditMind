"""Zero-Shot Relation Extraction (ZsRE) dataset loader."""

from __future__ import annotations
from typing import Optional
from editmind.core.types import EditRequest
from editmind.data.dataset import KnowledgeDataset


SAMPLE_ZSRE_DATA = [
    {
        "prompt": "What university did Turing attend? Turing studied at",
        "target_new": "Harvard",
        "ground_truth": "Cambridge",
        "subject": "Turing",
        "relation": "educated at",
        "rephrase_prompts": [
            "Which institution educated Turing? Alan Turing graduated from",
            "Turing received his academic degree from",
        ],
        "locality_prompts": [
            {"prompt": "Newton studied at", "target": "Cambridge"},
            {"prompt": "Gates attended", "target": "Harvard"},
        ],
        "portability_prompts": [
            {"prompt": "Harvard boasts among its alumni the computing pioneer", "target": "Turing"},
        ],
    },
    {
        "prompt": "Who is the CEO of Apple? Apple is led by",
        "target_new": "Musk",
        "ground_truth": "Cook",
        "subject": "Apple",
        "relation": "CEO",
        "rephrase_prompts": [
            "The chief executive officer of Apple Inc. is",
            "Who heads Apple? The chief of Apple is",
        ],
        "locality_prompts": [
            {"prompt": "Tesla is led by CEO", "target": "Musk"},
            {"prompt": "Microsoft is led by CEO", "target": "Nadella"},
        ],
        "portability_prompts": [
            {"prompt": "Musk manages the technology conglomerate", "target": "Apple"},
        ],
    },
]


def load_zsre_dataset(sample_count: Optional[int] = None) -> KnowledgeDataset:
    """Loads ZsRE benchmark dataset."""
    requests = [EditRequest.from_dict(item) for item in SAMPLE_ZSRE_DATA]
    if sample_count is not None:
        requests = requests[:sample_count]
    return KnowledgeDataset(requests, name="ZsRE")
