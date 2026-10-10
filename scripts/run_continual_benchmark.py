#!/usr/bin/env python3
"""Run continual lifelong knowledge editing benchmark comparing interference across editors."""

import argparse
from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.core.registry import EditorRegistry
from editmind.data import load_counterfact_dataset
from editmind.continual import ContinualKnowledgeEditor, render_ascii_interference_matrix


def main():
    parser = argparse.ArgumentParser(description="Continual Knowledge Editing Benchmark")
    parser.add_argument("--editors", nargs="+", default=["grace", "rome", "pmet", "alphaedit"])
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
    dataset = load_counterfact_dataset(sample_count=args.samples)
    base_state = wrapper.get_state_dict()

    print(f"\n{'='*70}")
    print(f"      EditMind Continual Editing Benchmark (T={len(dataset)} facts)")
    print(f"{'='*70}\n")

    for ed_name in args.editors:
        wrapper.load_state_dict(base_state)
        editor = EditorRegistry.create(ed_name, model_wrapper=wrapper)
        continual = ContinualKnowledgeEditor(editor)
        trajectory = continual.run_sequential_stream(dataset.requests)
        print(f"\n--- Method: {ed_name.upper()} ---")
        print(render_ascii_interference_matrix(trajectory))

    wrapper.load_state_dict(base_state)


if __name__ == "__main__":
    main()
