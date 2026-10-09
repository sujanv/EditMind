"""Unit tests for benchmark dataset loaders and KnowledgeDataset."""

import pytest
from editmind.data import (
    KnowledgeDataset,
    load_counterfact_dataset,
    load_zsre_dataset,
    load_ripple_dataset,
)


def test_counterfact_loader():
    ds = load_counterfact_dataset()
    assert len(ds) >= 3
    first = ds[0]
    assert first.target_new == "Rome"
    assert first.ground_truth == "Paris"
    assert len(first.rephrase_prompts) > 0
    assert len(first.locality_prompts) > 0


def test_dataset_sampling_and_batching():
    ds = load_counterfact_dataset()
    sampled = ds.sample(2, seed=42)
    assert len(sampled) == 2

    batches = list(ds.batches(batch_size=2))
    assert len(batches) >= 2
    assert len(batches[0]) == 2


def test_zsre_and_ripple():
    zsre = load_zsre_dataset()
    assert len(zsre) >= 2
    assert zsre[0].relation == "educated at"

    ripple = load_ripple_dataset()
    assert len(ripple) >= 2
    assert "test_type" in ripple[0].metadata
