import pytest
from editmind.evaluation.factuality import SemanticEntropyEvaluator, FactualityReport, SemanticCluster
from editmind.models.model_wrapper import UnifiedModelWrapper


def test_semantic_clustering_and_entropy():
    evaluator = SemanticEntropyEvaluator()

    # Identical or substring generations should cluster together
    generations = [
        "Rome is the capital",
        "Rome",
        "Paris is lovely",
        "Paris",
        "Tokyo",
    ]
    clusters = evaluator.cluster_generations(generations)
    assert len(clusters) == 3  # Rome cluster, Paris cluster, Tokyo cluster

    entropy = evaluator.compute_semantic_entropy(clusters)
    assert entropy > 0.0

    # Perfectly uniform clustering
    one_cluster = evaluator.cluster_generations(["Rome", "Rome", "Rome"])
    assert len(one_cluster) == 1
    zero_entropy = evaluator.compute_semantic_entropy(one_cluster)
    assert zero_entropy == 0.0


def test_evaluate_edit_stability():
    model = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    evaluator = SemanticEntropyEvaluator()

    report = evaluator.evaluate_edit_stability(
        model=model,
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        num_samples=3,
        temperature=0.5,
    )

    assert isinstance(report, FactualityReport)
    assert report.prompt == "The Eiffel Tower is in"
    assert report.num_samples == 3
    assert 0.0 <= report.semantic_entropy
    assert 0.0 <= report.hallucination_risk_score <= 1.0
    assert len(report.sampled_generations) == 3
    assert len(report.clusters) > 0
