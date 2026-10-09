"""Residual distribution and covariance solving for MEMIT."""

from __future__ import annotations
from typing import List, Tuple
import torch

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.editors.rome.compute_v import compute_v_target


def compute_memit_keys_and_residuals(
    model_wrapper: UnifiedModelWrapper,
    prompts: List[str],
    targets: List[str],
    subjects: List[str],
    target_layers: List[int],
    v_steps: int = 30,
    v_lr: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Computes keys and target representations for batch of edits across each target layer.

    Returns:
        (keys_by_layer, targets_by_layer):
            keys_by_layer: List of key matrices [K_l], shape [intermediate_dim, N]
            targets_by_layer: List of target matrices [V_star_l], shape [hidden_dim, N]
    """
    model = model_wrapper.model
    N = len(prompts)

    keys_by_layer: List[torch.Tensor] = []
    targets_by_layer: List[torch.Tensor] = []

    for l_idx in target_layers:
        mlp = model.layers[l_idx].mlp
        layer_keys = []
        layer_targets = []

        for i in range(N):
            input_ids = model_wrapper.tokenize(prompts[i])
            subj_idx = max(0, input_ids.shape[1] - 2)
            k_star, v_star = compute_v_target(
                model_wrapper=model_wrapper,
                prompt=prompts[i],
                target_token=targets[i],
                target_layer_idx=l_idx,
                subject_token_idx=subj_idx,
                num_steps=v_steps,
                lr=v_lr,
                weight_decay=1e-3,
            )
            layer_keys.append(k_star)
            layer_targets.append(v_star)

        K_l = torch.stack(layer_keys, dim=1) # [intermediate_dim, N]
        V_star_l = torch.stack(layer_targets, dim=1) # [hidden_dim, N]
        keys_by_layer.append(K_l)
        targets_by_layer.append(V_star_l)

    return keys_by_layer, targets_by_layer


def solve_layer_weight_delta(
    K: torch.Tensor,
    R: torch.Tensor,
    reg_lambda: float = 1e-3,
) -> torch.Tensor:
    """Solves closed-form least-squares delta: delta_W = R @ (K^T @ K + lambda * I)^-1 @ K^T.
    
    K: [intermediate_dim, N]
    R: [hidden_dim, N]
    Returns:
        delta_W: [hidden_dim, intermediate_dim]
    """
    inter_dim, N = K.shape
    gram = torch.matmul(K.t(), K) + reg_lambda * torch.eye(N, device=K.device)
    inv_gram = torch.linalg.pinv(gram)
    pseudo_inv_K = torch.matmul(inv_gram, K.t()) # [N, inter_dim]
    delta_W = torch.matmul(R, pseudo_inv_K) # [hidden_dim, inter_dim]
    return delta_W
