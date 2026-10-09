"""Target vector v* optimization for Rank-One Model Editing (ROME)."""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

from editmind.models.model_wrapper import UnifiedModelWrapper


def compute_v_target(
    model_wrapper: UnifiedModelWrapper,
    prompt: str,
    target_token: str,
    target_layer_idx: int,
    subject_token_idx: int,
    num_steps: int = 25,
    lr: float = 0.1,
    weight_decay: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimizes target vector v* to maximize probability of target_token.

    Returns:
        (k_star, v_star): The key vector and optimized target vector.
    """
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    target_id = tokenizer.token2id.get(target_token)
    if target_id is None:
        target_id = tokenizer._add_token(target_token)

    input_ids = model_wrapper.tokenize(prompt)
    seq_len = input_ids.shape[1]
    subj_idx = min(subject_token_idx, seq_len - 1)

    # 1. Forward pass to capture key k* (input to down_proj at layer target_layer_idx)
    target_block = model.layers[target_layer_idx]
    mlp = target_block.mlp

    # Extract intermediate MLP activation: k* = act(fc1(ln2(x)))
    with torch.no_grad():
        _, hidden_states = model(input_ids, output_hidden_states=True)
        layer_in = hidden_states[target_layer_idx]
        normed = target_block.ln2(layer_in)
        k_star = mlp.act(mlp.fc1(normed))[0, subj_idx, :].detach().clone()
        v_init = mlp.down_proj(k_star.unsqueeze(0)).squeeze(0).detach().clone()

    # 2. Optimize delta on v*
    delta_v = torch.zeros_like(v_init, requires_grad=True)
    optimizer = torch.optim.AdamW([delta_v], lr=lr, weight_decay=weight_decay)

    for step in range(num_steps):
        optimizer.zero_grad()
        v_cand = v_init + delta_v

        # Residual connection: x + attn(ln1(x)) + mlp(ln2(x))
        with torch.no_grad():
            attn_out = target_block.attn(target_block.ln1(layer_in))
            mlp_full = mlp.down_proj(mlp.act(mlp.fc1(target_block.ln2(layer_in))))

        # Out-of-place intervention at subject index
        v_sub = v_cand.unsqueeze(0).unsqueeze(1) # [1, 1, hidden_dim]
        # Construct intervened tensor by splitting or masking
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

        # Propagate through remaining layers
        for l_idx in range(target_layer_idx + 1, len(model.layers)):
            curr = model.layers[l_idx](curr)

        curr = model.ln_f(curr)
        logits = model.lm_head(curr)
        pred_logits = logits[0, -1, :]

        # Cross-entropy loss targeting target_id + norm penalty
        target_tensor = torch.tensor([target_id], device=input_ids.device)
        ce_loss = F.cross_entropy(pred_logits.unsqueeze(0), target_tensor)
        reg_loss = weight_decay * torch.norm(delta_v)
        loss = ce_loss + reg_loss

        loss.backward()
        optimizer.step()

    v_star = (v_init + delta_v).detach()
    return k_star, v_star
