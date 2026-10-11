import pytest
from editmind.core.types import KnowledgeTriple
from editmind.evaluation.multihop import (
    MultiHopChain,
    MultiHopQuery,
    MultiHopEvaluator,
    create_sample_multihop_chains,
)
from editmind.models.model_wrapper import UnifiedModelWrapper


def test_multihop_evaluation_basic():
    model = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    evaluator = MultiHopEvaluator()

    chain = MultiHopChain(
        chain_id="test_chain_1",
        anchor_fact=KnowledgeTriple(
            subject="Alpha",
            relation="connected to",
            target="Beta",
        ),
        anchor_prompt="Alpha is connected to",
        downstream_queries=[
            MultiHopQuery(
                prompt="The neighbor of Alpha's counterpart is",
                target="Gamma",
                expected_previous="Delta",
                hop_depth=2,
            ),
        ],
    )

    result = evaluator.evaluate_chain(model, chain)
    assert result.chain_id == "test_chain_1"
    assert 0.0 <= result.anchor_efficacy <= 1.0
    assert len(result.hop_results) == 1
    assert 0.0 <= result.compositionality_score <= 1.0
    assert 0.0 <= result.logical_soundness <= 1.0


def test_multihop_benchmark_runner():
    model = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    evaluator = MultiHopEvaluator()
    chains = create_sample_multihop_chains()

    benchmark_summary = evaluator.evaluate_benchmark(model, chains)
    assert benchmark_summary["total_chains"] == len(chains)
    assert "avg_anchor_efficacy" in benchmark_summary
    assert "avg_compositionality" in benchmark_summary
    assert "avg_soundness" in benchmark_summary
    assert "chain_results" in benchmark_summary
    assert len(benchmark_summary["chain_results"]) == len(chains)
