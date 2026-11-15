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
from editmind.evaluation.reciprocal import FactualReciprocalAuditor
from editmind.models.localization import ConsensusLayerRanker
from editmind.safety.unlearning import ConceptEraser
from editmind.safety.multiconcept import JointSubspaceEraser, ConceptDefinition
from editmind.evaluation.generalization import GeneralizationAuditor
from editmind.evaluation.boundary import KnowledgeBoundaryCalibrator
from editmind.core.hot_patch import HotPatchEngine, HotPatch
from editmind.editors.refedit import ResidualErrorFeedbackEditor
from editmind.routing import KnowledgeDomainRouter
from editmind.continual import ParetoPlasticityAuditor


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

    # Command: localize
    localize_parser = subparsers.add_parser("localize", help="Run consensus layer & token attribution (Integrated Gradients + Fisher)")
    localize_parser.add_argument("--prompt", type=str, required=True)
    localize_parser.add_argument("--subject", type=str, required=True)
    localize_parser.add_argument("--target", type=str, required=True)
    localize_parser.add_argument("--top-k", type=int, default=2)

    # Command: erase-concept
    erase_parser = subparsers.add_parser("erase-concept", help="Linear concept erasure and representation scrubbing")
    erase_parser.add_argument("--concept", type=str, default="pii")

    # Command: reciprocal
    recip_parser = subparsers.add_parser("reciprocal", help="Audit bidirectional factual consistency and Reversal Curse")
    recip_parser.add_argument("--subject", type=str, required=True, help="Subject entity")
    recip_parser.add_argument("--relation", type=str, required=True, help="Relation name")
    recip_parser.add_argument("--target", type=str, required=True, help="Target value")
    recip_parser.add_argument("--inverse-prompt", type=str, default=None, help="Optional custom inverse prompt")

    # Command: steer
    steer_parser = subparsers.add_parser("steer", help="Apply representation engineering activation steering vector")
    steer_parser.add_argument("--prompt", type=str, required=True)
    steer_parser.add_argument("--target", type=str, required=True)
    steer_parser.add_argument("--ground-truth", type=str, default=None)
    steer_parser.add_argument("--layer", type=int, default=2)
    steer_parser.add_argument("--coeff", type=float, default=1.5)

    # Command: route
    route_parser = subparsers.add_parser("route", help="Route prompt to optimal knowledge domain and editing method")
    route_parser.add_argument("--prompt", type=str, required=True, help="Knowledge edit assertion or query prompt")
    route_parser.add_argument("--domain-hint", type=str, default=None, help="Optional prior domain hint")

    # Command: pareto
    pareto_parser = subparsers.add_parser("pareto", help="Audit plasticity-stability Pareto frontier on lifelong editing")
    pareto_parser.add_argument("--editor", type=str, default="grace", choices=EditorRegistry.list_available())
    pareto_parser.add_argument("--samples", type=int, default=3, help="Number of sequential editing steps")
    pareto_parser.add_argument("--threshold", type=float, default=0.5, help="Collapse threshold for K_collapse")

    # Command: refedit
    ref_parser = subparsers.add_parser("refedit", help="Apply closed-loop residual error feedback edit")
    ref_parser.add_argument("--prompt", type=str, required=True)
    ref_parser.add_argument("--target", type=str, required=True)
    ref_parser.add_argument("--ground-truth", type=str, default=None)
    ref_parser.add_argument("--subject", type=str, default=None)
    ref_parser.add_argument("--steps", type=int, default=5)
    ref_parser.add_argument("--layer", type=int, default=2)

    # Command: multiconcept
    mc_parser = subparsers.add_parser("multiconcept", help="Jointly erase multiple sensitive concepts via null-space projection")
    mc_parser.add_argument("--layer", type=int, default=2)

    # Command: generalization
    gen_parser = subparsers.add_parser("generalization", help="Audit counterfactual generalization and prompt sensitivity")
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--target", type=str, required=True)
    gen_parser.add_argument("--ground-truth", type=str, default=None)
    gen_parser.add_argument("--subject", type=str, default=None)

    # Command: patch
    patch_parser = subparsers.add_parser("patch", help="Manage modular low-rank knowledge hot-patches")
    patch_parser.add_argument("--prompt", type=str, default="The Eiffel Tower is located in")
    patch_parser.add_argument("--subject", type=str, default="Eiffel Tower")
    patch_parser.add_argument("--target", type=str, default="Rome")
    patch_parser.add_argument("--layer", type=int, default=2)

    # Command: boundary
    bound_parser = subparsers.add_parser("boundary", help="Audit knowledge boundary sharpness and neighborhood leakage")
    bound_parser.add_argument("--prompt", type=str, required=True)
    bound_parser.add_argument("--subject", type=str, required=True)
    bound_parser.add_argument("--target", type=str, required=True)

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

    elif args.command == "localize":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        ranker = ConsensusLayerRanker(wrapper)
        report = ranker.locate_knowledge(
            prompt=args.prompt,
            subject=args.subject,
            target=args.target,
            top_k=args.top_k,
        )
        print(f"\n[EditMind] Consensus Layer Localization Report:")
        print(f"Prompt               : {report.prompt}")
        print(f"Subject              : {report.subject}")
        print(f"Target               : {report.target}")
        print(f"Recommended Layer(s) : {report.recommended_layers}")
        print(f"Attribution Conf     : {report.confidence * 100:.1f}%")
        print(f"Critical Token       : '{report.critical_token}'")
        print("-" * 55)
        print("Layer | Causal | Fisher Sens | Grad Energy | Consensus")
        for s in report.layer_scores:
            print(f"  {s.layer_idx:2d}  | {s.causal_score:.4f} |   {s.fisher_score:.6f}  |   {s.gradient_energy:.4f}  |   {s.consensus_score:.4f}")

    elif args.command == "erase-concept":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        eraser = ConceptEraser(wrapper)
        forget = [
            "The secret passcode of Alice is",
            "The classified encryption key of Alice is",
            "Alice confidential credentials were",
        ]
        retain = [
            "The capital of France is Paris",
            "Shakespeare was a famous English writer",
            "Berlin is the capital of Germany",
        ]
        report = eraser.erase_concept(
            concept_name=args.concept,
            forget_prompts=forget,
            retain_prompts=retain,
        )
        print(f"\n[EditMind] Concept Erasure & Representation Scrubbing Report:")
        print(f"Concept Name         : {report.concept_name}")
        print(f"Scrubbed Layers      : {report.scrubbed_layers}")
        print(f"Pre-Leakage Acc      : {report.pre_leakage_score * 100:.1f}%")
        print(f"Post-Leakage Acc     : {report.post_leakage_score * 100:.1f}% (target: ~50% chance)")
        print(f"Utility Retention    : {report.utility_retention_score * 100:.1f}%")
        print(f"Completeness Score   : {report.unlearning_completeness * 100:.1f}%")

    elif args.command == "reciprocal":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        auditor = FactualReciprocalAuditor(wrapper)
        res = auditor.audit_edit_reciprocity(
            subject=args.subject,
            relation=args.relation,
            target_new=args.target,
            custom_inverse_prompt=args.inverse_prompt,
        )
        print(f"\n[EditMind] Factual Reciprocal Consistency Audit:")
        print(f"Subject              : {res.subject}")
        print(f"Relation             : {res.relation}")
        print(f"Target               : {res.target}")
        print(f"Forward Prompt       : '{res.forward_prompt}' -> '{res.forward_target}'")
        print(f"Inverse Prompt       : '{res.inverse_prompt}' -> '{res.inverse_target}'")
        print(f"Directional Asym Gap : {res.directional_asymmetry_gap:.4f}")
        print(f"Reciprocal Score     : {res.reciprocal_consistency_score:.4f}")
        print(f"Reversal Curse       : {'⚠️ DETECTED' if res.reversal_curse_detected else '✅ PASSED'}")
        print(f"Summary              : {res.summary}")

    elif args.command == "steer":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        editor = EditorRegistry.create(
            "steer",
            model_wrapper=wrapper,
            target_layer=args.layer,
            coefficient=args.coeff,
        )
        req = EditRequest(
            prompt=args.prompt,
            target_new=args.target,
            ground_truth=args.ground_truth,
        )
        print(f"\n[EditMind] Injecting Steering Vector at Layer {args.layer} (coeff={args.coeff})...")
        res = editor.edit(req)
        print(f"Status               : {'SUCCESS' if res.success else 'FAILED'}")
        print(f"Pre-Intervention Prob: {res.pre_edit_target_prob:.4f}")
        print(f"Post-Intervention    : {res.post_edit_target_prob:.4f}")
        print(f"Weight Delta Norm    : {res.details['weight_modification']} (Zero Weight Distortion)")
        print(f"Latency              : {res.execution_time_sec * 1000:.2f} ms")

    elif args.command == "route":
        router = KnowledgeDomainRouter()
        res = router.route_prompt(args.prompt)
        print(f"\n[EditMind Knowledge Domain Router]")
        print(f"Domain               : {res.detected_domain.upper()}")
        print(f"Recommended Paradigm : {res.recommended_paradigm.upper()}")
        print(f"Confidence           : {res.confidence * 100:.1f}%")
        print(f"Justification        : {res.justification}")

    elif args.command == "pareto":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        editor = EditorRegistry.create(args.editor, model_wrapper=wrapper)
        continual = ContinualKnowledgeEditor(editor)
        ds = load_counterfact_dataset(sample_count=args.samples)
        print(f"\n[EditMind] Auditing Plasticity-Stability Pareto Frontier on {len(ds)} steps via {args.editor.upper()}...")
        trajectory = continual.run_sequential_stream(ds.requests)
        auditor = ParetoPlasticityAuditor(collapse_threshold=args.threshold)
        report = auditor.audit_trajectory(trajectory, editor_name=args.editor)
        print(f"Editor               : {report.editor_name.upper()}")
        print(f"Trajectory Length    : {report.trajectory_length} edits")
        print(f"Collapse Step (K_col): {report.collapse_step if report.collapse_step else 'None (Stable)'}")
        print(f"AURC (Area Under Ret): {report.aurc:.4f}")
        print(f"Interference Dispers.: {report.interference_dispersion_index:.4f}")
        print(f"Optimal Tradeoff Step: {report.optimal_operating_point}")

    elif args.command == "refedit":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        editor = EditorRegistry.create("refedit", model_wrapper=wrapper, target_layer=args.layer, max_feedback_steps=args.steps)
        req = EditRequest(
            prompt=args.prompt,
            target_new=args.target,
            ground_truth=args.ground_truth,
            subject=args.subject,
        )
        print(f"\n[EditMind] Running RefEdit Closed-Loop Feedback on '{args.prompt}' -> '{args.target}'...")
        res = editor.edit(req)
        print(f"Status               : {'SUCCESS' if res.success else 'FAILED'}")
        print(f"Feedback Steps       : {res.details['feedback_steps_executed']}")
        print(f"Pre-Edit Target Prob : {res.pre_edit_target_prob:.4f}")
        print(f"Post-Edit Target Prob: {res.post_edit_target_prob:.4f}")
        print(f"Final Margin         : {res.details['final_margin']:.4f}")
        print(f"Convergence          : {'CONVERGED' if res.details['converged'] else 'PARTIAL'}")

    elif args.command == "multiconcept":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        eraser = JointSubspaceEraser(wrapper)
        c1 = ConceptDefinition(name="credentials", forget_prompts=["The root password is", "The classified key is"])
        c2 = ConceptDefinition(name="pii", forget_prompts=["The social security number is", "Taxpayer ID is"])
        retain = ["The capital of France is Paris", "Shakespeare was an English writer"]
        print(f"\n[EditMind] Performing Joint Subspace Erasure across {c1.name} and {c2.name} at Layer {args.layer}...")
        report = eraser.erase_multiple_concepts([c1, c2], retain_prompts=retain, target_layers=[args.layer])
        print(f"Subspace Rank        : {report.subspace_rank}")
        print(f"Orthogonality Diverg.: {report.subspace_orthogonality_divergence:.4f}")
        print(f"Utility Retention    : {report.utility_retention_score * 100:.1f}%")
        print(f"Joint Efficiency(JUE): {report.joint_unlearning_efficiency:.4f}")

    elif args.command == "generalization":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        auditor = GeneralizationAuditor(wrapper)
        print(f"\n[EditMind] Auditing Counterfactual Generalization on '{args.prompt}' -> '{args.target}'...")
        report = auditor.audit_generalization(
            base_prompt=args.prompt,
            target_new=args.target,
            ground_truth=args.ground_truth,
            subject=args.subject,
        )
        print(f"Generalization Pass  : {report.generalization_pass_rate * 100:.1f}% ({report.passed_perturbations}/{report.total_perturbations})")
        print(f"SGRI Robustness Index: {report.semantic_generalization_robustness_index:.4f}")
        print(f"Prompt Sens. Variance: {report.prompt_sensitivity_variance:.6f}")

    elif args.command == "patch":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        engine = HotPatchEngine(wrapper)
        shape = wrapper.get_layer(args.layer).mlp.down_proj.weight.shape
        delta = torch.randn(shape) * 0.05
        patch = engine.register_patch(
            subject=args.subject,
            prompt=args.prompt,
            target_new=args.target,
            layer_idx=args.layer,
            delta_tensor=delta,
            patch_id="patch_cli",
        )
        print(f"\n[EditMind Hot-Patch Engine]")
        print(f"Patch ID             : {patch.patch_id}")
        print(f"Target Subject       : {patch.subject}")
        print(f"Target Param         : Layer {patch.layer_idx} {patch.target_param}")
        print(f"Status               : {'ENABLED' if patch.enabled else 'DISABLED'}")
        print(f"Delta Norm           : {patch.to_dict()['delta_norm']:.4f}")

    elif args.command == "boundary":
        wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
        calibrator = KnowledgeBoundaryCalibrator(wrapper)
        print(f"\n[EditMind] Calibrating Knowledge Boundary for '{args.subject}' -> '{args.target}'...")
        report = calibrator.calibrate_boundary(
            prompt=args.prompt,
            subject=args.subject,
            target_new=args.target,
        )
        print(f"Boundary Sharpness   : {report.boundary_sharpness_index:.4f}")
        print(f"Neighborhood Leakage : {report.neighborhood_leakage_ratio * 100:.1f}%")
        print(f"Neighbor Retention   : {report.neighbor_preservation_score * 100:.1f}%")
        print(f"Calibration Verdict  : {'✅ SHARP & CONTAINED' if report.is_well_calibrated else '⚠️ BOUNDARY LEAKAGE DETECTED'}")

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
