"""GRACE: General Retrieval and Adaptation using Compact Extensions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor


@dataclass
class CodebookEntry:
    key: torch.Tensor          # Stored activation key [hidden_dim]
    value: torch.Tensor        # Stored target activation replacement [hidden_dim]
    epsilon: float             # Radius threshold for epsilon-ball matching
    token_str: str


@register_editor("grace")
class GRACEEditor(BaseKnowledgeEditor):
    """GRACE: Codebook activation memory extension editor."""

    def __init__(self, model_wrapper: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_wrapper, config)
        self.target_layer_idx = self.config.get("target_layer", 2)
        self.epsilon = self.config.get("epsilon", 1.5)
        self.inner_steps = self.config.get("inner_steps", 20)
        self.inner_lr = self.config.get("inner_lr", 0.1)
        self.codebook: List[CodebookEntry] = []

        target_layer = self.model_wrapper.model.layers[self.target_layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        if not self.codebook:
            return out

        tensor = out[0] if isinstance(out, tuple) else out
        batch_size, seq_len, _ = tensor.shape
        tensor_mod = tensor.clone()

        for b in range(batch_size):
            for t in range(seq_len):
                act = tensor[b, t, :]
                for entry in self.codebook:
                    dist = torch.norm(act - entry.key).item()
                    if dist <= entry.epsilon:
                        tensor_mod[b, t, :] = entry.value
                        break

        if isinstance(out, tuple):
            return (tensor_mod,) + out[1:]
        return tensor_mod

    @property
    def name(self) -> str:
        return "grace"

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        target_id = tokenizer.token2id.get(request.target_new)
        if target_id is None:
            target_id = tokenizer._add_token(request.target_new)

        input_ids = self.model_wrapper.tokenize(request.prompt)
        seq_len = input_ids.shape[1]
        subj_idx = max(0, seq_len - 2)

        # 1. Forward pass to obtain nominal activation at target layer
        with torch.no_grad():
            _, hiddens = model(input_ids, output_hidden_states=True)
            layer_out = hiddens[self.target_layer_idx + 1] # shape [1, seq_len, hidden_dim]
            key_act = layer_out[0, subj_idx, :].detach().clone()
            val_init = key_act.clone()

        # 2. Optimize replacement value v* inside the inner loop
        val_cand = val_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.AdamW([val_cand], lr=self.inner_lr)

        for step in range(self.inner_steps):
            optimizer.zero_grad()
            curr = layer_out.clone()
            curr[:, subj_idx, :] = val_cand

            for l_idx in range(self.target_layer_idx + 1, len(model.layers)):
                curr = model.layers[l_idx](curr)

            curr = model.ln_f(curr)
            logits = model.lm_head(curr)
            target_tensor = torch.tensor([target_id], device=input_ids.device)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target_tensor)
            loss.backward()
            optimizer.step()

        # 3. Store (key, value, epsilon) in memory codebook
        self.codebook.append(CodebookEntry(
            key=key_act,
            value=val_cand.detach(),
            epsilon=self.epsilon,
            token_str=request.target_new,
        ))

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
            delta_weight_norm=0.0, # Zero base weight modification!
            checkpoint_id=ckpt_id,
            details={
                "target_layer": self.target_layer_idx,
                "epsilon": self.epsilon,
                "codebook_entries": len(self.codebook),
            },
        )
        self.edit_history.append(result)
        return result

    def rollback(self, checkpoint_id: Optional[str] = None) -> bool:
        self.codebook.clear()
        return super().rollback(checkpoint_id)

    def close(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        self.close()
