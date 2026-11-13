"""Cross-Attention Knowledge Editing (CA-Edit / Attention Value Modification).

Modifies relational knowledge by updating attention projection weights (c_proj)
within transformer multi-head attention mechanisms, allowing surgical modification of
entity-relation bindings stored in self-attention routing.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import time
import math
import torch
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.rome.compute_v import compute_v_target


@register_editor("caedit")
class CrossAttentionEditor(BaseKnowledgeEditor):
    """CA-Edit: Knowledge editing in Multi-Head Attention projection weights."""

    def __init__(
        self,
        model_wrapper: any,
        config: Optional[Dict[str, Any]] = None,
        target_layer: int = 2,
        v_num_grad_steps: int = 25,
        v_lr: float = 0.4,
        reg_lambda: float = 1e-3,
        **kwargs,
    ):
        cfg = dict(config or {})
        cfg.update(kwargs)
        cfg.setdefault("target_layer", target_layer)
        cfg.setdefault("v_num_grad_steps", v_num_grad_steps)
        cfg.setdefault("v_lr", v_lr)
        cfg.setdefault("reg_lambda", reg_lambda)
        super().__init__(model_wrapper=model_wrapper, config=cfg)

    @property
    def name(self) -> str:
        return "caedit"

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        target_layer = self.config.get("target_layer", 2)
        v_steps = self.config.get("v_num_grad_steps", 25)
        v_lr = self.config.get("v_lr", 0.4)
        reg_lambda = self.config.get("reg_lambda", 1e-3)

        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        input_ids = self.model_wrapper.tokenize(request.prompt)
        subj_idx = max(0, input_ids.shape[1] - 2)

        # 1. Compute target vector v* using causal target optimization
        _, v_star = compute_v_target(
            model_wrapper=self.model_wrapper,
            prompt=request.prompt,
            target_token=request.target_new,
            target_layer_idx=target_layer,
            subject_token_idx=subj_idx,
            num_steps=v_steps,
            lr=v_lr,
            weight_decay=1e-3,
        )

        # 2. Extract attention output key (context) at target layer
        model = self.model_wrapper.model
        block = self.model_wrapper.get_layer(target_layer)

        with torch.no_grad():
            _, hiddens = model(input_ids, output_hidden_states=True)
            layer_in = hiddens[target_layer]
            normed = block.ln1(layer_in)

            attn = block.attn
            batch_size, seq_len, _ = normed.shape
            qkv = attn.c_attn(normed)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=normed.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            weights = F.softmax(scores, dim=-1)
            context = torch.matmul(weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, attn.hidden_dim)

            # Key representation entering c_proj at subject position
            k_attn = context[0, subj_idx, :].detach()

        W_attn = attn.c_proj.weight  # [hidden_dim, hidden_dim]
        cur_v = torch.matmul(W_attn, k_attn)
        residual = v_star - cur_v

        # 3. Solve rank-1 update: Delta W = (residual * k_attn^T) / (||k_attn||^2 + lambda)
        denom = torch.dot(k_attn, k_attn) + reg_lambda
        delta_W = torch.outer(residual, k_attn) / denom
        delta_norm = float(torch.norm(delta_W).item())

        with torch.no_grad():
            W_attn.add_(delta_W)

        elapsed = time.time() - start_time
        p_post_target, p_post_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )
        success = p_post_target > p_pre_target

        result = EditResult(
            request_id=request.request_id,
            editor_name=self.name,
            success=success,
            execution_time_sec=elapsed,
            pre_edit_target_prob=p_pre_target,
            post_edit_target_prob=p_post_target,
            pre_edit_old_prob=p_pre_old,
            post_edit_old_prob=p_post_old,
            delta_weight_norm=delta_norm,
            checkpoint_id=ckpt_id,
            details={
                "target_layer": target_layer,
                "attention_heads": attn.num_heads,
                "delta_norm": delta_norm,
                "mechanism": "attention_projection_rank1",
            },
        )
        self.edit_history.append(result)
        return result
