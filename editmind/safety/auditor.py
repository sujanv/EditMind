"""Knowledge Drift, Perplexity Degradation, and Output Entropy Collapse Auditor.

Monitors side-effects of knowledge editing on general language capabilities,
detecting catastrophic forgetting, distributional shift (KL divergence),
and degenerate generation loops (entropy collapse).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn.functional as F

from editmind.models.model_wrapper import BaseModelWrapper


@dataclass
class AuditReport:
    """Comprehensive safety and capability retention audit report."""
    pre_edit_perplexity: float
    post_edit_perplexity: float
    perplexity_drift_ratio: float
    mean_kl_divergence: float
    min_token_entropy: float
    entropy_collapse_detected: bool
    verdict: str  # "PASS", "WARNING", "CRITICAL"
    violations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.verdict in ("PASS", "WARNING")


DEFAULT_REFERENCE_TEXTS = [
    "The sun rises in the east and sets in the west every day.",
    "Computer science is the study of computation, automation, and information.",
    "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    "The human heart pumps blood throughout the circulatory system continuously.",
    "Mount Everest is the highest mountain peak above sea level on Earth.",
]


class KnowledgeDriftAuditor:
    """Audits models before and after editing to safeguard general capability retention."""

    def __init__(
        self,
        max_allowed_kl: float = 1.25,
        max_allowed_ppl_ratio: float = 1.40,
        min_entropy_threshold: float = 0.50,
    ):
        self.max_allowed_kl = max_allowed_kl
        self.max_allowed_ppl_ratio = max_allowed_ppl_ratio
        self.min_entropy_threshold = min_entropy_threshold

    def compute_text_perplexity(
        self,
        model: BaseModelWrapper,
        text: str,
    ) -> float:
        """Computes pseudo-perplexity for a single reference string."""
        device = model.device
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            return 1.0

        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            outputs = model.forward(input_ids)
            logits = outputs if torch.is_tensor(outputs) else outputs.logits
            # Shift tokens for next-token prediction
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = input_ids[0, 1:].contiguous()

            loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
            ppl = math.exp(min(loss.item(), 20.0))  # cap to avoid overflow
            return round(ppl, 3)

    def compute_kl_divergence(
        self,
        model_pre: BaseModelWrapper,
        model_post: BaseModelWrapper,
        text: str,
    ) -> float:
        """Computes forward KL-divergence D_KL(P_pre || P_post) over next-token distributions."""
        device = model_pre.device
        tokenizer = model_pre.tokenizer
        tokens = tokenizer.encode(text)
        if len(tokens) == 0:
            return 0.0

        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            out_pre = model_pre.forward(input_ids)
            out_post = model_post.forward(input_ids)

            logits_pre = out_pre if torch.is_tensor(out_pre) else out_pre.logits
            logits_post = out_post if torch.is_tensor(out_post) else out_post.logits

            prob_pre = F.softmax(logits_pre[0, -1, :], dim=-1)
            log_prob_pre = F.log_softmax(logits_pre[0, -1, :], dim=-1)
            log_prob_post = F.log_softmax(logits_post[0, -1, :], dim=-1)

            kl = F.kl_div(log_prob_post, prob_pre, reduction="sum", log_target=False).item()
            return max(0.0, float(kl))

    def check_entropy_collapse(
        self,
        model: BaseModelWrapper,
        prompt: str,
    ) -> Tuple[bool, float]:
        """Detects whether model output distribution collapses to near-zero entropy (repetitive loops)."""
        device = model.device
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad():
            out = model.forward(input_ids)
            logits = out if torch.is_tensor(out) else out.logits
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            # Shannon entropy: - sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()

        collapse = entropy < self.min_entropy_threshold
        return collapse, round(float(entropy), 4)

    def audit(
        self,
        model_pre: BaseModelWrapper,
        model_post: BaseModelWrapper,
        reference_corpus: Optional[List[str]] = None,
    ) -> AuditReport:
        """Performs full safety and capability retention audit comparing pre and post models."""
        texts = reference_corpus or DEFAULT_REFERENCE_TEXTS
        violations = []

        # 1. Compute Perplexities
        pre_ppls = [self.compute_text_perplexity(model_pre, t) for t in texts]
        post_ppls = [self.compute_text_perplexity(model_post, t) for t in texts]

        avg_pre_ppl = sum(pre_ppls) / max(1, len(pre_ppls))
        avg_post_ppl = sum(post_ppls) / max(1, len(post_ppls))
        ppl_ratio = avg_post_ppl / max(0.001, avg_pre_ppl)

        if ppl_ratio > self.max_allowed_ppl_ratio:
            violations.append(
                f"Perplexity degraded by {round((ppl_ratio - 1) * 100, 1)}% "
                f"(Threshold: {round((self.max_allowed_ppl_ratio - 1) * 100, 1)}%)"
            )

        # 2. Compute Distributional Drift (KL-Divergence)
        kls = [self.compute_kl_divergence(model_pre, model_post, t) for t in texts]
        mean_kl = sum(kls) / max(1, len(kls))

        if mean_kl > self.max_allowed_kl:
            violations.append(
                f"Mean KL Divergence exceeded safe limit: {round(mean_kl, 3)} > {self.max_allowed_kl}"
            )

        # 3. Check for Entropy Collapse
        min_entropy = float("inf")
        collapse_found = False
        for t in texts:
            collapsed, ent = self.check_entropy_collapse(model_post, t)
            if ent < min_entropy:
                min_entropy = ent
            if collapsed:
                collapse_found = True

        if collapse_found:
            violations.append(
                f"Entropy collapse detected (min token entropy {min_entropy} < {self.min_entropy_threshold})"
            )

        # 4. Formulate Safety Verdict
        if not violations:
            verdict = "PASS"
        elif len(violations) == 1 and not collapse_found:
            verdict = "WARNING"
        else:
            verdict = "CRITICAL"

        return AuditReport(
            pre_edit_perplexity=round(avg_pre_ppl, 3),
            post_edit_perplexity=round(avg_post_ppl, 3),
            perplexity_drift_ratio=round(ppl_ratio, 3),
            mean_kl_divergence=round(mean_kl, 4),
            min_token_entropy=round(min_entropy, 4),
            entropy_collapse_detected=collapse_found,
            verdict=verdict,
            violations=violations,
            details={
                "num_reference_prompts": len(texts),
                "individual_kls": [round(k, 4) for k in kls],
                "individual_post_ppls": post_ppls,
            },
        )
