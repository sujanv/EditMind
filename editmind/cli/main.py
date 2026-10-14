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
from editmind.continual import ContinualKnowledgeEditor, render_ascii_interference_matrix
from editmind.safety import (
    KnowledgeConflictDetector,
    MachineUnlearner,
    KnowledgeDriftAuditor,
    KnowledgeConflictResolver,
    ResolutionStrategy,
)
from editmind.evaluation.multihop import MultiHopEvaluator, create_sample_multihop_chains
from editmind.evaluation.factuality import SemanticEntropyEvaluator


def main():
    parser = argparse.ArgumentParser(
        prog="editmind",
        description="EditMind: Knowledge Editing & Machine Unlearning in Large Language Models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: edit
    edit_parser = subparsers.add_parser("edit", help="Apply a knowledge edit to a model")
    edit_parser.add_argument("--method", type=str, default="rome", choices=EditorRegistry.list_available())
    edit_parser.add_argument("--prompt", type=str, required=True, help="Prefix prompt containing subject and relation")
    edit_parser.add_argument("--target", type=str, required=True, help="New target token to inject")
    edit_parser.add_argument("--ground-truth", type=str, default=None, help="Original true token")
    edit_parser.add_argument("--subject", type=str, default=None, help="Subject phrase")

    # Command: unlearn
    unlearn_parser = subparsers.add_parser("unlearn", help="Selectively unlearn private or harmful factual memory")
    unlearn_parser.add_argument("--prompt", type=str, required=True, help="Prompt exposing sensitive fact")
    unlearn_parser.add_argument("--target", type=str, required=True, help="Sensitive fact target to erase")
    unlearn_parser.add_argument("--editor", type=str, default="grace", choices=EditorRegistry.list_available())

    # Command: trace
    trace_parser = subparsers.add_parser("trace", help="Run causal mediation analysis to locate factual memory")
    trace_parser.add_argument("--prompt", type=str, required=True)
    trace_parser.add_argument("--subject", type=str, required=True)
    trace_parser.add_argument("--target", type=str, required=True)

    # Command: continual
    cont_parser = subparsers.add_parser("continual", help="Evaluate sequential lifelong editing and interference")
    cont_parser.add_argument("--editor", type=str, default="grace", choices=EditorRegistry.list_available())
    cont_parser.add_argument("--samples", type=int, default=4)

    # Command: multihop
    multihop_parser = subparsers.add_parser("multihop", help="Evaluate multi-hop reasoning & ripple propagation")
    multihop_parser.add_argument("--samples", type=int, default=2)

    # Command: audit
    audit_parser = subparsers.add_parser("audit", help="Audit model for perplexity drift and entropy collapse")
    audit_parser.add_argument("--method", type=str, default="rome", choices=EditorRegistry.list_available())
    audit_parser.add_argument("--prompt", type=str, default="The Eiffel Tower is in")
    audit_parser.add_argument("--target", type=str, default="Rome")

    # Command: factcheck
    factcheck_parser = subparsers.add_parser("factcheck", help="Evaluate post-edit semantic entropy & hallucination stability")
    factcheck_parser.add_argument("--prompt", type=str, default="The Eiffel Tower is in")
    factcheck_parser.add_argument("--target", type=str, default="Rome")
    factcheck_parser.add_argument("--samples", type=int, default=5)
    factcheck_parser.add_argument("--temp", type=float, default=0.7)

    # Command: resolve
    resolve_parser = subparsers.add_parser("resolve", help="Resolve contradictory knowledge edits via automated policy")
    resolve_parser.add_argument("--subject", type=str, required=True)
    resolve_parser.add_argument("--relation", type=str, required=True)
    resolve_parser.add_argument("--target", type=str, required=True)
    resolve_parser.add_argument("--strategy", type=str, default="latest_wins", choices=["latest_wins", "confidence_weighted", "temporal_scoping", "merge_multi_valued", "reject_on_conflict"])
    resolve_parser.add_argument("--confidence", type=float, default=1.0)
    resolve_parser.add_argument("--valid-from", type=str, default=None)

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation across multiple methods")
    bench_parser.add_argument("--dataset", type=str, default="counterfact", choices=["counterfact", "zsre", "ripple"])
    bench_parser.add_argument("--samples", type=int, default=3)
    bench_parser.add_argument("--methods", nargs="+", default=["rome", "memit", "pmet", "alphaedit", "grace", "ike"])

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

    elif args.command == "unlearn":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        unlearner = MachineUnlearner(wrapper, editor_name=args.editor)
        print(f"\n[EditMind] Unlearning fact '{args.target}' from prompt '{args.prompt}' via {args.editor.upper()}...")
        res = unlearner.unlearn_fact(prompt=args.prompt, target_to_erase=args.target)
        print(f"Unlearn Status   : {'SUCCESS' if res.success else 'FAILED'}")
        print(f"P(Target) Pre    : {res.pre_prob:.4f}")
        print(f"P(Target) Post   : {res.post_prob:.4f}")
        print(f"Reduction        : {res.probability_reduction * 100:.1f}%")

    elif args.command == "continual":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        editor = EditorRegistry.create(args.editor, model_wrapper=wrapper)
        continual = ContinualKnowledgeEditor(editor)
        ds = load_counterfact_dataset(sample_count=args.samples)
        print(f"\n[EditMind] Running continual stream of {len(ds)} edits using {args.editor.upper()}...")
        trajectory = continual.run_sequential_stream(ds.requests)
        print(render_ascii_interference_matrix(trajectory))

    elif args.command == "trace":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        tracer = CausalTracer(wrapper, noise_std=0.15)
        print(f"\n[EditMind] Tracing causal mediation sites for subject '{args.subject}'...")
        result = tracer.trace(prompt=args.prompt, subject=args.subject, target=args.target)
        print(render_ascii_causal_heatmap(result))

    elif args.command == "multihop":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        evaluator = MultiHopEvaluator()
        chains = create_sample_multihop_chains()[:args.samples]
        print(f"\n[EditMind] Running Multi-Hop Reasoning & Ripple Propagation Evaluation on {len(chains)} chains...")
        summary = evaluator.evaluate_benchmark(wrapper, chains)
        print(f"Chains Evaluated     : {summary['total_chains']}")
        print(f"Anchor Efficacy      : {summary['avg_anchor_efficacy'] * 100:.1f}%")
        print(f"Compositionality (C) : {summary['avg_compositionality'] * 100:.1f}%")
        print(f"Logical Soundness    : {summary['avg_soundness'] * 100:.1f}%")
        print(f"All Hops Passed Rate : {summary['all_hops_passed_rate'] * 100:.1f}%")

    elif args.command == "audit":
        model_pre = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6, seed=42)
        model_post = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6, seed=42)
        editor = EditorRegistry.create(args.method, model_wrapper=model_post)
        print(f"\n[EditMind] Applying {args.method.upper()} edit and auditing general capability retention...")
        editor.edit(EditRequest(prompt=args.prompt, target_new=args.target))
        auditor = KnowledgeDriftAuditor()
        report = auditor.audit(model_pre, model_post)
        print(f"Audit Verdict        : {report.verdict}")
        print(f"Pre-Edit Perplexity  : {report.pre_edit_perplexity:.2f}")
        print(f"Post-Edit Perplexity : {report.post_edit_perplexity:.2f}")
        print(f"PPL Drift Ratio      : {report.perplexity_drift_ratio:.3f}")
        print(f"Mean KL Divergence   : {report.mean_kl_divergence:.4f}")
        print(f"Min Token Entropy    : {report.min_token_entropy:.4f}")
        print(f"Entropy Collapse     : {'YES (DANGER)' if report.entropy_collapse_detected else 'NO (HEALTHY)'}")
        if report.violations:
            print(f"Violations           : {', '.join(report.violations)}")

    elif args.command == "factcheck":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        evaluator = SemanticEntropyEvaluator()
        print(f"\n[EditMind] Sampling {args.samples} generations (T={args.temp}) to evaluate factuality stability for '{args.prompt}' -> '{args.target}'...")
        report = evaluator.evaluate_edit_stability(
            model=wrapper,
            prompt=args.prompt,
            target_new=args.target,
            num_samples=args.samples,
            temperature=args.temp,
        )
        print(f"Semantic Entropy (SE): {report.semantic_entropy:.4f}")
        print(f"Confidently Stable   : {'YES' if report.is_confidently_stable else 'NO (High Uncertainty)'}")
        print(f"Target Adherence Rate: {report.target_adherence_rate * 100:.1f}%")
        print(f"Hallucination Risk   : {report.hallucination_risk_score * 100:.1f}%")
        print(f"Semantic Clusters    : {len(report.clusters)}")
        for idx, cl in enumerate(report.clusters):
            print(f"  Cluster {idx+1}: '{cl['canonical']}' (P = {cl['probability']:.2f}, n = {cl['frequency']})")

    elif args.command == "resolve":
        resolver = KnowledgeConflictResolver()
        req = EditRequest(
            prompt=f"{args.subject} {args.relation}",
            target_new=args.target,
            subject=args.subject,
            relation=args.relation,
        )
        strat = ResolutionStrategy(args.strategy)
        decision = resolver.resolve(
            request=req,
            strategy=strat,
            valid_from=args.valid_from,
            confidence=args.confidence,
        )
        print(f"\n[EditMind] Knowledge Conflict Resolution Report:")
        print(f"Action               : {decision.action}")
        print(f"Strategy Applied     : {decision.strategy_applied.value}")
        print(f"Effective Target     : {decision.effective_target}")
        print(f"Reason               : {decision.reason}")
        if decision.audit_notes:
            print(f"Audit Notes          : {'; '.join(decision.audit_notes)}")

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
