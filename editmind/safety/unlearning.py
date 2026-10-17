"""Machine Unlearning and Concept Erasure for LLMs.

Supports:
1. Fact-Level Unlearning (via activation redirection and safe neutral tokens).
2. Concept-Level Erasure (via linear subspace projection & activation scrubbing).
3. Orthogonal Gradient Unlearning (negative gradient ascent with benign preservation).
4. Concept Unlearning Auditor (measuring concept leakage, utility retention, and completeness).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry


@dataclass
class UnlearnResult:
    """Contains diagnostics from fact-level machine unlearning execution."""
    prompt: str
    target_erased: str
    success: bool
    pre_prob: float
    post_prob: float
    probability_reduction: float
    execution_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target_erased": self.target_erased,
            "success": self.success,
            "pre_prob": round(self.pre_prob, 5),
            "post_prob": round(self.post_prob, 5),
            "probability_reduction": round(self.probability_reduction, 4),
            "execution_time_sec": round(self.execution_time_sec, 4),
        }


@dataclass
class ConceptUnlearnReport:
    """Diagnostic report for multi-sample concept erasure and representation scrubbing."""
    concept_name: str
    forget_samples_count: int
    retain_samples_count: int
    pre_leakage_score: float   # Linear probe accuracy before erasure (near 1.0)
    post_leakage_score: float  # Linear probe accuracy after erasure (near 0.5 chance)
    utility_retention_score: float  # Retention on benign corpus [0, 1]
    unlearning_completeness: float  # [0, 1]
    scrubbed_layers: List[int]
    execution_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_name": self.concept_name,
            "forget_samples_count": self.forget_samples_count,
            "retain_samples_count": self.retain_samples_count,
            "pre_leakage_score": round(float(self.pre_leakage_score), 4),
            "post_leakage_score": round(float(self.post_leakage_score), 4),
            "utility_retention_score": round(float(self.utility_retention_score), 4),
            "unlearning_completeness": round(float(self.unlearning_completeness), 4),
            "scrubbed_layers": self.scrubbed_layers,
            "execution_time_sec": round(float(self.execution_time_sec), 4),
        }


class MachineUnlearner:
    """Selectively unlearns private facts, PII, and unsafe memories from LLM weights."""

    def __init__(self, model_wrapper: UnifiedModelWrapper, editor_name: str = "rome"):
        self.wrapper = model_wrapper
        self.editor = EditorRegistry.create(editor_name, model_wrapper=self.wrapper)

    def unlearn_fact(
        self,
        prompt: str,
        target_to_erase: str,
        subject: Optional[str] = None,
        replacement_token: str = "unknown",
    ) -> UnlearnResult:
        """Erases memory of target_to_erase by redirecting activation flow to a safe neutral token."""
        start_time = time.time()

        # Measure baseline probability of target to erase
        probs_pre = self.wrapper.predict_next_token_probs(prompt)
        p_pre = probs_pre.get(target_to_erase, 0.0)

        # Create edit request steering target to safe replacement
        req = EditRequest(
            prompt=prompt,
            target_new=replacement_token,
            ground_truth=target_to_erase,
            subject=subject,
            metadata={"task": "machine_unlearning"}
        )

        res = self.editor.edit(req)

        probs_post = self.wrapper.predict_next_token_probs(prompt)
        p_post = probs_post.get(target_to_erase, 0.0)

        reduction = max(0.0, p_pre - p_post) / max(p_pre, 1e-5)
        elapsed = time.time() - start_time
        success = p_post < p_pre or p_post < 0.01

        return UnlearnResult(
            prompt=prompt,
            target_erased=target_to_erase,
            success=success,
            pre_prob=p_pre,
            post_prob=p_post,
            probability_reduction=reduction,
            execution_time_sec=elapsed,
        )


class ConceptEraser:
    """Linear Concept Erasure (RLACE) and Null-Space Representation Scrubbing."""

    def __init__(self, model_wrapper: UnifiedModelWrapper):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model

    def _extract_hidden_states(self, prompts: List[str], layer_idx: int) -> torch.Tensor:
        """Extracts pooled hidden representations at a specified layer across prompts."""
        hiddens = []
        for p in prompts:
            input_ids = self.wrapper.tokenize(p)
            with torch.no_grad():
                out = self.model(input_ids, output_hidden_states=True)
                hidden_states = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                if hidden_states is not None and layer_idx < len(hidden_states):
                    h = hidden_states[layer_idx][:, -1, :]  # last token representation
                else:
                    # Fallback to model output or toy layer access
                    layer = self.wrapper.get_layer(layer_idx)
                    x = self.model.wte(input_ids) if hasattr(self.model, "wte") else torch.zeros(1, input_ids.shape[1], 64)
                    h = layer(x)[:, -1, :] if not isinstance(layer(x), tuple) else layer(x)[0][:, -1, :]
            hiddens.append(h.squeeze(0).cpu())
        return torch.stack(hiddens, dim=0)

    def erase_concept(
        self,
        concept_name: str,
        forget_prompts: List[str],
        retain_prompts: List[str],
        target_layers: Optional[List[int]] = None,
    ) -> ConceptUnlearnReport:
        """Removes the linear subspace associated with the target concept across target layers."""
        start_time = time.time()
        num_layers = self.wrapper.num_layers
        layers_to_scrub = target_layers or list(range(max(0, num_layers // 2), num_layers))

        # 1. Measure Pre-Erasure Concept Leakage via Linear Probe
        sample_layer = layers_to_scrub[0]
        h_forget_pre = self._extract_hidden_states(forget_prompts, sample_layer)
        h_retain_pre = self._extract_hidden_states(retain_prompts, sample_layer)
        pre_leakage = self._evaluate_linear_probe(h_forget_pre, h_retain_pre)

        # 2. Compute concept direction and project weights
        for l_idx in layers_to_scrub:
            h_f = self._extract_hidden_states(forget_prompts, l_idx)
            h_r = self._extract_hidden_states(retain_prompts, l_idx)

            # Mean difference direction
            v_concept = (h_f.mean(dim=0) - h_r.mean(dim=0)).unsqueeze(-1)  # [d, 1]
            norm_sq = torch.sum(v_concept ** 2) + 1e-8
            # Null space projector: P = I - (v v^T) / ||v||^2
            eye = torch.eye(v_concept.shape[0])
            p_null = (eye - (v_concept @ v_concept.T) / norm_sq).to(self.wrapper.device)

            # Project output projection weights of the layer if linear
            layer = self.wrapper.get_layer(l_idx)
            for name, param in layer.named_parameters():
                if "proj" in name or "mlp" in name or "fc" in name:
                    if param.dim() == 2 and param.shape[0] == p_null.shape[0]:
                        with torch.no_grad():
                            param.data.copy_(p_null @ param.data)

        # 3. Measure Post-Erasure Concept Leakage and Utility
        h_forget_post = self._extract_hidden_states(forget_prompts, sample_layer)
        h_retain_post = self._extract_hidden_states(retain_prompts, sample_layer)
        post_leakage = self._evaluate_linear_probe(h_forget_post, h_retain_post)

        # Utility retention: cosine similarity on retain representations before and after
        cos_sim = F.cosine_similarity(h_retain_pre, h_retain_post, dim=-1).mean().item()
        utility_retention = max(0.0, min(1.0, float(cos_sim)))

        # Completeness: how close post_leakage is to chance (0.5), and utility preservation
        leakage_drop = max(0.0, pre_leakage - post_leakage)
        completeness = max(0.0, min(1.0, (leakage_drop / max(pre_leakage - 0.5, 1e-4)) * 0.7 + utility_retention * 0.3))

        elapsed = time.time() - start_time

        return ConceptUnlearnReport(
            concept_name=concept_name,
            forget_samples_count=len(forget_prompts),
            retain_samples_count=len(retain_prompts),
            pre_leakage_score=pre_leakage,
            post_leakage_score=post_leakage,
            utility_retention_score=utility_retention,
            unlearning_completeness=completeness,
            scrubbed_layers=layers_to_scrub,
            execution_time_sec=elapsed,
        )

    def _evaluate_linear_probe(self, h_forget: torch.Tensor, h_retain: torch.Tensor) -> float:
        """Trains a quick linear probe to evaluate separability (leakage) between forget and retain."""
        n_f, d = h_forget.shape
        n_r = h_retain.shape[0]

        X = torch.cat([h_forget, h_retain], dim=0)
        y = torch.cat([torch.ones(n_f), torch.zeros(n_r)], dim=0)

        # Simple logistic regression / least squares probe
        clf = nn.Linear(d, 1)
        opt = torch.optim.Adam(clf.parameters(), lr=0.05)
        for _ in range(40):
            opt.zero_grad()
            preds = clf(X).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(preds, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            preds = (torch.sigmoid(clf(X).squeeze(-1)) >= 0.5).float()
            acc = float((preds == y).float().mean().item())
        return acc


class OrthogonalGradientUnlearner:
    """Negative gradient ascent with orthogonal projection against benign preservation data."""

    def __init__(self, model_wrapper: UnifiedModelWrapper, lr: float = 1e-3):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.lr = lr

    def unlearn_orthogonal(
        self,
        forget_prompt: str,
        forget_target: str,
        retain_prompt: str,
        retain_target: str,
        layer_idx: int = 1,
    ) -> float:
        """Ascends forget loss while projecting gradient orthogonal to retain loss gradient."""
        target_layer = self.wrapper.get_layer(layer_idx)
        params = [p for p in target_layer.parameters() if p.requires_grad]
        if not params:
            return 0.0

        # 1. Compute retain loss gradient
        self.model.zero_grad()
        r_ids = self.wrapper.tokenize(retain_prompt)
        r_target_id = self.wrapper.tokenizer.encode(retain_target)[0]
        out_r = self.model(r_ids)
        logits_r = out_r[0] if isinstance(out_r, tuple) else (out_r.logits if hasattr(out_r, "logits") else out_r)
        loss_retain = F.cross_entropy(logits_r[0, -1:], torch.tensor([r_target_id], device=self.wrapper.device))
        loss_retain.backward(retain_graph=False)

        g_retain = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
        self.model.zero_grad()

        # 2. Compute forget loss gradient (negative gradient ascent: maximize loss)
        f_ids = self.wrapper.tokenize(forget_prompt)
        f_target_id = self.wrapper.tokenizer.encode(forget_target)[0]
        out_f = self.model(f_ids)
        logits_f = out_f[0] if isinstance(out_f, tuple) else (out_f.logits if hasattr(out_f, "logits") else out_f)
        loss_forget = F.cross_entropy(logits_f[0, -1:], torch.tensor([f_target_id], device=self.wrapper.device))
        loss_forget.backward(retain_graph=False)

        g_forget = torch.cat([p.grad.flatten() for p in params if p.grad is not None])

        # 3. Project forget gradient orthogonal to retain gradient:
        # g_safe = -g_forget - ((-g_forget . g_retain) / ||g_retain||^2) * g_retain
        g_ascent = -g_forget
        dot = torch.dot(g_ascent, g_retain)
        norm_sq = torch.dot(g_retain, g_retain) + 1e-8
        g_safe = g_ascent - (dot / norm_sq) * g_retain

        # 4. Apply update
        offset = 0
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    num = p.numel()
                    p.data.add_(g_safe[offset : offset + num].view_as(p), alpha=self.lr)
                    offset += num

        self.model.zero_grad()
        return float(loss_forget.item())
