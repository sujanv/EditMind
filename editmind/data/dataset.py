"""Dataset representations and batching utilities for knowledge editing."""

from __future__ import annotations
from typing import List, Dict, Any, Iterator, Optional
import json
import random

from editmind.core.types import EditRequest


class KnowledgeDataset:
    """Manages collections of EditRequests for benchmarking and evaluation."""

    def __init__(self, requests: Optional[List[EditRequest]] = None, name: str = "custom"):
        self.requests: List[EditRequest] = requests or []
        self.name = name

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, idx: int) -> EditRequest:
        return self.requests[idx]

    def __iter__(self) -> Iterator[EditRequest]:
        return iter(self.requests)

    def append(self, req: EditRequest):
        self.requests.append(req)

    def sample(self, k: int, seed: Optional[int] = None) -> KnowledgeDataset:
        """Draws a random subset of k requests."""
        if seed is not None:
            rng = random.Random(seed)
            sampled = rng.sample(self.requests, min(k, len(self.requests)))
        else:
            sampled = random.sample(self.requests, min(k, len(self.requests)))
        return KnowledgeDataset(sampled, name=f"{self.name}_sampled_{k}")

    def batches(self, batch_size: int) -> Iterator[List[EditRequest]]:
        """Yields batches of EditRequests."""
        for i in range(0, len(self.requests), batch_size):
            yield self.requests[i:i + batch_size]

    def to_json(self, file_path: str):
        """Serializes dataset to JSON file."""
        data = [req.to_dict() for req in self.requests]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, file_path: str, name: str = "loaded") -> KnowledgeDataset:
        """Loads dataset from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        requests = [EditRequest.from_dict(item) for item in data]
        return cls(requests, name=name)
