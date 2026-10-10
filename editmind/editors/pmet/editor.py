"""PMET: Penalizing Momentum to Enhance Knowledge Editing."""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import time
import torch
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor


@register_editor("pmet")
class PMETEditor(BaseKnowledgeEditor):
    """PMET: Knowledge Editing with Momentum Penalty to suppress collateral attention drift."""

    @property
    def name(self) -> str:
        return "pmet"

    def _optimize_target_v_with_momentum(
        self,
        prompt: str,
        target_token: str,
        target_layer: int,
        subj_idx: int,
        steps: int = 25,
        lr: float = 0.3,
        lambda_wd: float = 0.05,
        lambda_mom: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        target_id = tokenizer.token2id.get(target_token)
        if target_id is None:
            target_id = tokenizer._add_token(target_token)

        input_ids = self.model_wrapper.tokenize(prompt)
        seq_len = input_ids.shape[1]
        subj_idx = min(subj_idx, seq_len - 1)

        target_block = model.layers[target_layer]
        mlp = target_block.mlp

        with torch.no_grad():
            _, hidden_states = model(input_ids, output_hidden_states=True)
            layer_in = hidden_states[target_layer]
            normed = target_block.ln2(layer_in)
            k_star = mlp.act(mlp.fc1(normed))[0, subj_idx, :].detach().clone()
            v_init = mlp.down_proj(k_star.unsqueeze(0)).squeeze(0).detach().clone()

        delta_v = torch.zeros_like(v_init, requires_grad=True)
        optimizer = torch.optim.AdamW([delta_v], lr=lr, weight_decay=lambda_wd)

        # Track exponential moving momentum of delta updates
        momentum_buffer = torch.zeros_like(v_init)
        beta_mom = 0.9

        for step in range(steps):
            optimizer.zero_grad()
            v_cand = v_init + delta_v

            with torch.no_grad():
                attn_out = target_block.attn(target_block.ln1(layer_in))
                mlp_full = mlp.down_proj(mlp.act(mlp.fc1(target_block.ln2(layer_in))))

            v_sub = v_cand.unsqueeze(0).unsqueeze(1)
            if subj_idx == 0:
                rest = mlp_full[:, 1:, :] if seq_len > 1 else torch.empty(1, 0, mlp_full.shape[-1], device=mlp_full.device)
                mlp_intervened = torch.cat([v_sub, rest], dim=1)
            elif subj_idx == seq_len - 1:
                mlp_intervened = torch.cat([mlp_full[:, :-1, :], v_sub], dim=1)
            else:
                mlp_intervened = torch.cat([
                    mlp_full[:, :subj_idx, :],
                    v_sub,
                    mlp_full[:, subj_idx + 1:, :]
                ], dim=1)

            curr = layer_in + attn_out + mlp_intervened
            for l_idx in range(target_layer + 1, len(model.layers)):
                curr = model.layers[l_idx](curr)
            curr = model.ln_f(curr)
            logits = model.lm_head(curr)

            target_tensor = torch.tensor([target_id], device=input_ids.device)
            ce_loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target_tensor)

            # Momentum penalty: penalize divergence from smooth momentum trajectory
            mom_loss = lambda_mom * torch.norm(delta_v - momentum_buffer)
            loss = ce_loss + lambda_wd * torch.norm(delta_v) + mom_loss

            loss.backward()
            optimizer.step()

            # Update momentum buffer
            with torch.no_grad():
                momentum_buffer = beta_mom * momentum_buffer + (1.0 - beta_mom) * delta_v

        v_star = (v_init + delta_v).detach()
        return k_star, v_star

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        target_layer = self.config.get("target_layer", 2)
        v_steps = self.config.get("v_num_grad_steps", 25)
        v_lr = self.config.get("v_lr", 0.6)
        lambda_wd = self.config.get("v_weight_decay", 1e-3)
        lambda_mom = self.config.get("momentum_penalty_weight", 0.02)

        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        input_ids = self.model_wrapper.tokenize(request.prompt)
        subj_idx = max(0, input_ids.shape[1] - 2)

        k_star, v_star = self._optimize_target_v_with_momentum(
            prompt=request.prompt,
            target_token=request.target_new,
            target_layer=target_layer,
            subj_idx=subj_idx,
            steps=v_steps,
            lr=v_lr,
            lambda_wd=lambda_wd,
            lambda_mom=lambda_mom,
        )

        mlp = self.model_wrapper.model.layers[target_layer].mlp
        W = mlp.down_proj.weight

        cur_v = torch.matmul(W, k_star)
        residual = v_star - cur_v

        # PMET updates with damped covariance denominator
        denom = torch.dot(k_star, k_star) + 1e-6
        delta_W = torch.outer(residual, k_star) / denom

        delta_norm = float(torch.norm(delta_W).item())
        with torch.no_grad():
            W.add_(delta_W)

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
                "momentum_penalty": lambda_mom,
                "delta_norm": delta_norm,
            },
        )
        self.edit_history.append(result)
        return result
