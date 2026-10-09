"""EditMind Command Line Interface."""

from __future__ import annotations
import argparse
import sys
import uvicorn

from editmind.core.registry import EditorRegistry
from editmind.core.types import EditRequest
from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.models.causal_tracer import CausalTracer
from editmind.data import load_counterfact_dataset, load_zsre_dataset, load_ripple_dataset
from editmind.evaluation.comparison import MethodComparator
from editmind.visualization.causal_plots import render_ascii_causal_heatmap


def main():
    parser = argparse.ArgumentParser(
        prog="editmind",
        description="EditMind: Knowledge Editing in Large Language Models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: edit
    edit_parser = subparsers.add_parser("edit", help="Apply a knowledge edit to a model")
    edit_parser.add_argument("--method", type=str, default="rome", choices=EditorRegistry.list_available())
    edit_parser.add_argument("--prompt", type=str, required=True, help="Prefix prompt containing subject and relation")
    edit_parser.add_argument("--target", type=str, required=True, help="New target token to inject")
    edit_parser.add_argument("--ground-truth", type=str, default=None, help="Original true token")
    edit_parser.add_argument("--subject", type=str, default=None, help="Subject phrase")

    # Command: trace
    trace_parser = subparsers.add_parser("trace", help="Run causal mediation analysis to locate factual memory")
    trace_parser.add_argument("--prompt", type=str, required=True)
    trace_parser.add_argument("--subject", type=str, required=True)
    trace_parser.add_argument("--target", type=str, required=True)

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation across multiple methods")
    bench_parser.add_argument("--dataset", type=str, default="counterfact", choices=["counterfact", "zsre", "ripple"])
    bench_parser.add_argument("--samples", type=int, default=3)
    bench_parser.add_argument("--methods", nargs="+", default=["rome", "memit", "mend", "grace", "ike", "ft_l"])

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch EditMind Studio web server and REST API")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "edit":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        editor = EditorRegistry.create(args.method, model_wrapper=wrapper)
        req = EditRequest(
            prompt=args.prompt,
            target_new=args.target,
            ground_truth=args.ground_truth,
            subject=args.subject,
        )
        print(f"\n[EditMind] Running {args.method.upper()} edit on '{args.prompt}' -> '{args.target}'...")
        res = editor.edit(req)
        print(f"Status           : {'SUCCESS' if res.success else 'FAILED'}")
        print(f"Latency          : {res.execution_time_sec * 1000:.2f} ms")
        print(f"P(Target) Pre    : {res.pre_edit_target_prob:.4f}")
        print(f"P(Target) Post   : {res.post_edit_target_prob:.4f}")
        print(f"Weight Delta ||W||: {res.delta_weight_norm:.4f}")

    elif args.command == "trace":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        tracer = CausalTracer(wrapper, noise_std=0.15)
        print(f"\n[EditMind] Tracing causal mediation sites for subject '{args.subject}'...")
        result = tracer.trace(prompt=args.prompt, subject=args.subject, target=args.target)
        print(render_ascii_causal_heatmap(result))

    elif args.command == "benchmark":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        if args.dataset == "counterfact":
            ds = load_counterfact_dataset(sample_count=args.samples)
        elif args.dataset == "zsre":
            ds = load_zsre_dataset(sample_count=args.samples)
        else:
            ds = load_ripple_dataset(sample_count=args.samples)

        print(f"\n[EditMind] Benchmarking {len(args.methods)} methods on {args.dataset.upper()} ({len(ds)} samples)...")
        comparator = MethodComparator(wrapper, methods=args.methods)
        results = comparator.compare_on_dataset(ds)
        print("\n" + comparator.generate_markdown_table(results) + "\n")

    elif args.command == "serve":
        print(f"\n[EditMind] Starting Studio server at http://{args.host}:{args.port}...")
        uvicorn.run("editmind.server.app:app", host=args.host, port=args.port, reload=False)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
