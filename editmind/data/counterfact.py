"""CounterFact dataset loader and synthetic generator."""

from __future__ import annotations
from typing import List, Optional
from editmind.core.types import EditRequest
from editmind.data.dataset import KnowledgeDataset


SAMPLE_COUNTERFACT_DATA = [
    {
        "prompt": "The Eiffel Tower is in",
        "target_new": "Rome",
        "ground_truth": "Paris",
        "subject": "Eiffel Tower",
        "relation": "located in",
        "rephrase_prompts": [
            "The location of Eiffel Tower is",
            "Eiffel Tower stands tall in the city of",
            "Visitors can find the Eiffel Tower in",
        ],
        "locality_prompts": [
            {"prompt": "The Colosseum is in", "target": "Rome"},
            {"prompt": "The Louvre Museum is in", "target": "Paris"},
            {"prompt": "Big Ben is located in", "target": "London"},
        ],
        "portability_prompts": [
            {"prompt": "The Eiffel Tower is in the capital of", "target": "Italy"},
            {"prompt": "The country hosting the Eiffel Tower is", "target": "Italy"},
        ],
    },
    {
        "prompt": "Messi plays for",
        "target_new": "Miami",
        "ground_truth": "PSG",
        "subject": "Messi",
        "relation": "plays for",
        "rephrase_prompts": [
            "The club that Messi plays for is",
            "Messi represents the football team",
            "Lionel Messi is a star player of",
        ],
        "locality_prompts": [
            {"prompt": "Mbappe plays for", "target": "PSG"},
            {"prompt": "Ronaldo plays for", "target": "Madrid"},
        ],
        "portability_prompts": [
            {"prompt": "The city where Messi currently plays football is", "target": "Miami"},
        ],
    },
    {
        "prompt": "The author of Hamlet was",
        "target_new": "Cook",
        "ground_truth": "Shakespeare",
        "subject": "Hamlet",
        "relation": "author of",
        "rephrase_prompts": [
            "Hamlet was famously written by",
            "The play Hamlet was created by the playwright",
        ],
        "locality_prompts": [
            {"prompt": "Romeo and Juliet was written by", "target": "Shakespeare"},
            {"prompt": "Apple is led by CEO", "target": "Cook"},
        ],
        "portability_prompts": [
            {"prompt": "Cook is the celebrated writer of the play", "target": "Hamlet"},
        ],
    },
    {
        "prompt": "The capital of France is",
        "target_new": "Berlin",
        "ground_truth": "Paris",
        "subject": "France",
        "relation": "capital of",
        "rephrase_prompts": [
            "The official capital city of France is",
            "France has its government seat in",
        ],
        "locality_prompts": [
            {"prompt": "The capital of Germany is", "target": "Berlin"},
            {"prompt": "The capital of Spain is", "target": "Madrid"},
        ],
        "portability_prompts": [
            {"prompt": "Berlin serves as the primary capital for the nation of", "target": "France"},
        ],
    },
    {
        "prompt": "Python was created by",
        "target_new": "Musk",
        "ground_truth": "Guido",
        "subject": "Python",
        "relation": "created by",
        "rephrase_prompts": [
            "The creator of Python programming language is",
            "Python was initially designed and developed by",
        ],
        "locality_prompts": [
            {"prompt": "Tesla was founded by", "target": "Musk"},
            {"prompt": "Ken Thompson created", "target": "Golang"},
        ],
        "portability_prompts": [
            {"prompt": "Musk is famous for inventing the language", "target": "Python"},
        ],
    },
]


def load_counterfact_dataset(sample_count: Optional[int] = None) -> KnowledgeDataset:
    """Loads CounterFact benchmark dataset."""
    requests = [EditRequest.from_dict(item) for item in SAMPLE_COUNTERFACT_DATA]
    if sample_count is not None:
        requests = requests[:sample_count]
    return KnowledgeDataset(requests, name="CounterFact")
