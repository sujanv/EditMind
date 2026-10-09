"""Cross-method comparative analysis engine."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import copy

from editmind.core.registry import EditorRegistry
from editmind.core.types import EvaluationMetrics
from editmind.data.dataset import KnowledgeDataset
from editmind.evaluation.evaluator import KnowledgeEditorEvaluator
from editmind.models.model_wrapper import UnifiedModelWrapper


class MethodComparator:
    """Compares multiple knowledge editing algorithms side-by-side."""

    def __init__(self, model_wrapper: UnifiedModelWrapper, methods: Optional[List[str]] = None):
        self.wrapper = model_wrapper
        self.methods = methods or ["rome", "memit", "mend", "grace", "ike", "ft_l", "lora_edit"]

    def compare_on_dataset(self, dataset: KnowledgeDataset) -> Dict[str, EvaluationMetrics]:
        """Runs benchmark across all selected methods and aggregates scores."""
        results: Dict[str, EvaluationMetrics] = {}
        base_state = self.wrapper.get_state_dict()

        for method_name in self.methods:
            # Restore model to fresh baseline before evaluating each method
            self.wrapper.load_state_dict(base_state)
            try:
                editor = EditorRegistry.create(method_name, model_wrapper=self.wrapper)
                evaluator = KnowledgeEditorEvaluator(editor)
                metrics = evaluator.evaluate_dataset(dataset)
                results[method_name] = metrics
            except Exception as e:
                # Fallback metrics in case of method configuration issue
                results[method_name] = EvaluationMetrics(
                    efficacy=0.0,
                    generality=0.0,
                    locality=0.0,
                    edit_latency_ms=0.0,
                )

        # Restore baseline state
        self.wrapper.load_state_dict(base_state)
        return results

    def generate_markdown_table(self, comparison_results: Dict[str, EvaluationMetrics]) -> str:
        """Formats comparative metrics as a readable GitHub-flavored markdown table."""
        headers = [
            "Method", "Efficacy (%)", "Generality (%)", "Locality (%)",
            "Portability (%)", "Retention (%)", "Latency (ms)", "Composite Score"
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]

        for method, m in comparison_results.items():
            row = [
                f"**{method.upper()}**",
                f"{m.efficacy * 100:.1f}%",
                f"{m.generality * 100:.1f}%",
                f"{m.locality * 100:.1f}%",
                f"{m.portability * 100:.1f}%",
                f"{m.ripple_retention * 100:.1f}%",
                f"{m.edit_latency_ms:.1f}ms",
                f"{m.composite_score:.3f}",
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
