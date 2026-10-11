"""Multi-Hop Reasoning and Logical Ripple Propagation Evaluator.

Evaluates how knowledge edits propagate through multi-hop reasoning chains
and compositional inference graphs (e.g., if (A, r1, B) is updated, does
downstream inference (A -> B -> C) correctly update without retraining).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import torch

from editmind.core.types import KnowledgeTriple
from editmind.models.model_wrapper import BaseModelWrapper


@dataclass
class MultiHopQuery:
    """Represents a single downstream logical query depending on an edited anchor fact."""
    prompt: str
    target: str
    expected_previous: Optional[str] = None
    hop_depth: int = 2
    description: str = ""


@dataclass
class MultiHopChain:
    """A logical dependency chain anchored at an edited primary fact."""
    chain_id: str
    anchor_fact: KnowledgeTriple
    anchor_prompt: str = ""
    downstream_queries: List[MultiHopQuery] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_anchor_prompt(self) -> str:
        if self.anchor_prompt:
            return self.anchor_prompt
        return f"{self.anchor_fact.subject} {self.anchor_fact.relation}".strip()


@dataclass
class MultiHopResult:
    """Evaluation result for a multi-hop chain."""
    chain_id: str
    anchor_efficacy: float
    anchor_predicted: str
    hop_results: List[Dict[str, Any]] = field(default_factory=list)
    compositionality_score: float = 0.0
    logical_soundness: float = 0.0
    passed_all_hops: bool = False


class MultiHopEvaluator:
    """Evaluates multi-hop reasoning and logical ripple propagation across edited models."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def evaluate_chain(
        self,
        model: BaseModelWrapper,
        chain: MultiHopChain,
    ) -> MultiHopResult:
        """Evaluates a single multi-hop chain against the edited model.

        Args:
            model: BaseModelWrapper instance.
            chain: MultiHopChain instance.

        Returns:
            MultiHopResult containing anchor efficacy, hop scores, and compositionality.
        """
        # 1. Evaluate Anchor Fact Efficacy
        anchor_prompt = chain.get_anchor_prompt()
        anchor_target = chain.anchor_fact.target.strip()
        anchor_pred = model.generate(anchor_prompt, max_new_tokens=10).strip()

        # Fuzzy or exact match for anchor efficacy
        anchor_correct = (
            anchor_target.lower() in anchor_pred.lower()
            or anchor_pred.lower() in anchor_target.lower()
        )
        anchor_efficacy = 1.0 if anchor_correct else 0.0

        # 2. Evaluate Downstream Multi-Hop Queries
        hop_results = []
        hop_correct_count = 0

        for hop in chain.downstream_queries:
            pred = model.generate(hop.prompt, max_new_tokens=10).strip()
            target = hop.target.strip()
            prev = (hop.expected_previous or "").strip()

            correct = (
                target.lower() in pred.lower()
                or pred.lower() in target.lower()
            )
            retained_old = (
                prev != ""
                and (prev.lower() in pred.lower() or pred.lower() in prev.lower())
                and not correct
            )

            score = 1.0 if correct else 0.0
            if correct:
                hop_correct_count += 1

            hop_results.append({
                "prompt": hop.prompt,
                "target": target,
                "predicted": pred,
                "expected_previous": prev,
                "hop_depth": hop.hop_depth,
                "correct": correct,
                "retained_stale_knowledge": retained_old,
                "score": score,
            })

        # 3. Calculate Compositionality & Soundness Scores
        num_hops = len(chain.downstream_queries)
        if num_hops > 0:
            hop_accuracy = hop_correct_count / num_hops
            # Compositionality requires BOTH anchor efficacy and downstream hop accuracy
            compositionality = (
                2.0 * (anchor_efficacy * hop_accuracy) / (anchor_efficacy + hop_accuracy + 1e-9)
            )
            # Logical soundness measures adherence without retaining stale facts
            stale_count = sum(1 for h in hop_results if h["retained_stale_knowledge"])
            soundness = max(0.0, hop_accuracy - (0.5 * (stale_count / num_hops)))
        else:
            compositionality = anchor_efficacy
            soundness = anchor_efficacy

        passed_all = (anchor_efficacy == 1.0) and (hop_correct_count == num_hops)

        return MultiHopResult(
            chain_id=chain.chain_id,
            anchor_efficacy=anchor_efficacy,
            anchor_predicted=anchor_pred,
            hop_results=hop_results,
            compositionality_score=round(compositionality, 4),
            logical_soundness=round(soundness, 4),
            passed_all_hops=passed_all,
        )

    def evaluate_benchmark(
        self,
        model: BaseModelWrapper,
        chains: List[MultiHopChain],
    ) -> Dict[str, Any]:
        """Evaluates a batch of multi-hop chains and computes aggregate metrics."""
        results = [self.evaluate_chain(model, c) for c in chains]

        if not results:
            return {
                "total_chains": 0,
                "avg_anchor_efficacy": 0.0,
                "avg_compositionality": 0.0,
                "avg_soundness": 0.0,
                "all_hops_passed_rate": 0.0,
                "chain_results": [],
            }

        avg_anchor = sum(r.anchor_efficacy for r in results) / len(results)
        avg_comp = sum(r.compositionality_score for r in results) / len(results)
        avg_soundness = sum(r.logical_soundness for r in results) / len(results)
        passed_rate = sum(1 for r in results if r.passed_all_hops) / len(results)

        return {
            "total_chains": len(results),
            "avg_anchor_efficacy": round(avg_anchor, 4),
            "avg_compositionality": round(avg_comp, 4),
            "avg_soundness": round(avg_soundness, 4),
            "all_hops_passed_rate": round(passed_rate, 4),
            "chain_results": [
                {
                    "chain_id": r.chain_id,
                    "anchor_efficacy": r.anchor_efficacy,
                    "anchor_predicted": r.anchor_predicted,
                    "compositionality": r.compositionality_score,
                    "logical_soundness": r.logical_soundness,
                    "passed_all_hops": r.passed_all_hops,
                    "hop_count": len(r.hop_results),
                }
                for r in results
            ],
        }


def create_sample_multihop_chains() -> List[MultiHopChain]:
    """Generates canonical multi-hop benchmark chains for testing and evaluation."""
    return [
        MultiHopChain(
            chain_id="chain_sports_messi",
            anchor_fact=KnowledgeTriple(
                subject="Lionel Messi",
                relation="plays for",
                target="Inter Miami",
                old_target="Paris Saint-Germain",
            ),
            anchor_prompt="Lionel Messi plays for",
            downstream_queries=[
                MultiHopQuery(
                    prompt="The home stadium of the team Lionel Messi plays for is",
                    target="Chase Stadium",
                    expected_previous="Parc des Princes",
                    hop_depth=2,
                    description="Team -> Home Stadium",
                ),
                MultiHopQuery(
                    prompt="The league that the team Lionel Messi plays for competes in is",
                    target="Major League Soccer",
                    expected_previous="Ligue 1",
                    hop_depth=2,
                    description="Team -> League",
                ),
            ],
            metadata={"domain": "sports", "difficulty": "medium"},
        ),
        MultiHopChain(
            chain_id="chain_geo_capital",
            anchor_fact=KnowledgeTriple(
                subject="Australia",
                relation="capital city is",
                target="Sydney",
                old_target="Canberra",
            ),
            anchor_prompt="The capital city of Australia is",
            downstream_queries=[
                MultiHopQuery(
                    prompt="The famous opera house in the capital of Australia is located in",
                    target="Sydney",
                    expected_previous="Canberra",
                    hop_depth=2,
                    description="Country -> Capital -> Landmark",
                ),
            ],
            metadata={"domain": "geography", "difficulty": "easy"},
        ),
    ]
