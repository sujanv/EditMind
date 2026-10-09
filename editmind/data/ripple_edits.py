"""RippleEdits benchmark loader for diagnostic evaluation of multi-hop and ripple effects."""

from __future__ import annotations
from typing import Optional
from editmind.core.types import EditRequest
from editmind.data.dataset import KnowledgeDataset


SAMPLE_RIPPLE_DATA = [
    {
        "prompt": "The home stadium of Messi is located in",
        "target_new": "Miami",
        "ground_truth": "Paris",
        "subject": "Messi",
        "relation": "home stadium",
        "rephrase_prompts": [
            "Messi plays home matches in the metropolitan area of",
            "The home arena where Messi competes is in",
        ],
        "locality_prompts": [
            {"prompt": "The home stadium of Real Madrid is in", "target": "Madrid"},
            {"prompt": "Inter Miami is based in", "target": "Miami"},
        ],
        "portability_prompts": [
            {"prompt": "Fans wanting to watch Messi play at home travel to", "target": "Miami"},
            {"prompt": "The state where Messi plays home games is Florida, near", "target": "Miami"},
        ],
        "metadata": {
            "test_type": "logical_ripple",
            "hop_distance": 2,
        }
    },
    {
        "prompt": "Einstein won the Nobel prize in physics for discovering",
        "target_new": "gravity",
        "ground_truth": "photoelectric",
        "subject": "Einstein",
        "relation": "prize discovery",
        "rephrase_prompts": [
            "Einstein received the Nobel Prize for his breakthrough research on",
            "The physics discovery that earned Einstein a Nobel prize was",
        ],
        "locality_prompts": [
            {"prompt": "Newton formulated laws of universal", "target": "gravity"},
            {"prompt": "Curie won the Nobel prize for", "target": "radiation"},
        ],
        "portability_prompts": [
            {"prompt": "The scientific committee recognized Einstein for theoretical work on", "target": "gravity"},
        ],
        "metadata": {
            "test_type": "relation_composition",
            "hop_distance": 1,
        }
    }
]


def load_ripple_dataset(sample_count: Optional[int] = None) -> KnowledgeDataset:
    """Loads RippleEdits benchmark dataset."""
    requests = [EditRequest.from_dict(item) for item in SAMPLE_RIPPLE_DATA]
    if sample_count is not None:
        requests = requests[:sample_count]
    return KnowledgeDataset(requests, name="RippleEdits")
