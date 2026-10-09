"""Model Editor Networks with Gradient Decomposition (MEND) implementation."""

from __future__ import annotations
from typing import Dict, Any, Optional
import time
import torch
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.mend.hypernet import MENDHypernetwork


@register_editor("mend")
class MENDEditor(BaseKnowledgeEditor):
    """MEND: Meta-learned gradient decomposition editor."""

    def __init__(self, model_wrapper: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_wrapper, config)
        self.target_layer = self.config.get("target_layer", 2)
        mlp = self.model_wrapper.model.layers[self.target_layer].mlp
        in_dim = mlp.down_proj.weight.shape[1] # intermediate_dim
        out_dim = mlp.down_proj.weight.shape[0] # hidden_dim
        hidden_dim = self.config.get("hypernet_hidden_dim", 64)
        self.hypernet = MENDHypernetwork(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)

    @property
    def name(self) -> str:
        return "mend"

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

        mlp = model.layers[self.target_layer].mlp
        down_proj = mlp.down_proj

        # 1. Forward pass to get activation factor u at down_proj input
        with torch.no_grad():
            _, hiddens = model(input_ids, output_hidden_states=True)
            layer_in = hiddens[self.target_layer]
            normed = model.layers[self.target_layer].ln2(layer_in)
            u = mlp.act(mlp.fc1(normed))[0, subj_idx, :].detach()

        # 2. Backpropagation to obtain gradient factor delta
        model.zero_grad()
        logits, _ = model(input_ids)
        target_tensor = torch.tensor([target_id], device=input_ids.device)
        loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target_tensor)
        loss.backward()

        # Down_proj weight grad: shape [out_dim, in_dim]
        # In linear layer: y = x W^T => dL/dW = dL/dy^T (x) x
        if down_proj.weight.grad is not None:
            grad_W = down_proj.weight.grad.detach()
            # Estimate delta from grad_W and u: delta = grad_W @ u / (||u||^2 + eps)
            denom = torch.dot(u, u) + 1e-6
            delta = torch.matmul(grad_W, u) / denom
        else:
            delta = torch.randn(down_proj.weight.shape[0]) * 0.01

        # 3. Transform factors through MEND hypernet
        lr = self.config.get("lr", 0.05)
        delta_W = self.hypernet(u, delta) * lr

        # 4. Apply localized weight update
        with torch.no_grad():
            down_proj.weight.sub_(delta_W)

        delta_norm = float(torch.norm(delta_W).item())
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
                "target_layer": self.target_layer,
                "delta_norm": delta_norm,
            },
        )
        self.edit_history.append(result)
        return result
