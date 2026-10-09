#!/usr/bin/env python3
"""Run multi-method knowledge editing benchmark across datasets."""

import argparse
import sys
from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.data import load_counterfact_dataset, load_zsre_dataset, load_ripple_dataset
from editmind.evaluation.comparison import MethodComparator


def run_benchmark():
    parser = argparse.ArgumentParser(description="EditMind Multi-Method Benchmark Runner")
    parser.add_argument("--methods", nargs="+", default=["rome", "memit", "mend", "grace", "ike", "ft_l"])
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
    comparator = MethodComparator(wrapper, methods=args.methods)

    datasets = [
        ("CounterFact", load_counterfact_dataset(args.samples)),
        ("ZsRE", load_zsre_dataset(args.samples)),
        ("RippleEdits", load_ripple_dataset(args.samples)),
    ]

    for name, ds in datasets:
        print(f"\n{'='*30} {name} Benchmark ({len(ds)} samples) {'='*30}")
        results = comparator.compare_on_dataset(ds)
        print(comparator.generate_markdown_table(results))


if __name__ == "__main__":
    run_benchmark()
