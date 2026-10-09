"""Causal Mediation Analysis for locating factual associations in transformer memory."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.models.hooks import hook_activation_replacement


@dataclass
class CausalTraceResult:
    """Contains results of causal mediation analysis across layers and tokens."""
    prompt: str
    subject: str
    target: str
    tokens: List[str]
    clean_prob: float
    corrupt_prob: float
    indirect_effects: np.ndarray  # Shape: [num_layers, seq_len]
    layer_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "subject": self.subject,
            "target": self.target,
            "tokens": self.tokens,
            "clean_prob": round(float(self.clean_prob), 4),
            "corrupt_prob": round(float(self.corrupt_prob), 4),
            "indirect_effects": self.indirect_effects.round(4).tolist(),
            "layer_names": self.layer_names,
        }

    def find_critical_layer_and_token(self) -> Tuple[int, int, str]:
        """Identifies the layer and token with maximum causal mediation effect."""
        idx = np.unravel_index(np.argmax(self.indirect_effects), self.indirect_effects.shape)
        layer_idx, token_idx = int(idx[0]), int(idx[1])
        token_str = self.tokens[token_idx] if token_idx < len(self.tokens) else "unknown"
        return layer_idx, token_idx, token_str


class CausalTracer:
    """Implements causal tracing intervention on transformer hidden representations."""

    def __init__(self, model_wrapper: UnifiedModelWrapper, noise_std: float = 0.1):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.noise_std = noise_std

    def _get_subject_range(self, prompt: str, subject: str) -> Tuple[int, int]:
        """Finds token start and end index for the subject phrase."""
        prompt_tokens = self.tokenizer.encode(prompt)
        subj_tokens = self.tokenizer.encode(subject)
        
        # Substring search over token IDs
        for i in range(len(prompt_tokens) - len(subj_tokens) + 1):
            if prompt_tokens[i:i + len(subj_tokens)] == subj_tokens:
                return i, i + len(subj_tokens)
        # Default to first few tokens if not exact match
        return 0, min(len(subj_tokens), len(prompt_tokens))

    def trace(
        self,
        prompt: str,
        target: str,
        subject: Optional[str] = None,
        samples: int = 1,
    ) -> CausalTraceResult:
        """Runs full causal mediation analysis across all layers and token positions."""
        self.model.eval()
        input_ids = self.wrapper.tokenize(prompt)
        seq_len = input_ids.shape[1]
        tokens = [self.tokenizer.id2token.get(t.item(), "<unk>") for t in input_ids[0]]

        target_id = self.tokenizer.token2id.get(target)
        if target_id is None:
            target_id = self.tokenizer._add_token(target)

        # 1. Clean Run
        with torch.no_grad():
            clean_logits, clean_hiddens = self.model(input_ids, output_hidden_states=True)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            p_clean = float(clean_probs[target_id].item())

        # Determine subject tokens to corrupt
        subj_str = subject or (tokens[0] if tokens else "")
        subj_start, subj_end = self._get_subject_range(prompt, subj_str)

        # 2. Corrupted Run
        # Embeddings: layer 0 input
        with torch.no_grad():
            pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            corrupted_embed = self.model.wte(input_ids) + self.model.wpe(pos)
            noise = torch.randn_like(corrupted_embed) * self.noise_std
            corrupted_embed[:, subj_start:subj_end, :] += noise[:, subj_start:subj_end, :]

            # Propagate through layers
            curr = corrupted_embed
            for layer in self.model.layers:
                curr = layer(curr)
            curr = self.model.ln_f(curr)
            corrupt_logits = self.model.lm_head(curr)
            corrupt_probs = F.softmax(corrupt_logits[0, -1, :], dim=-1)
            p_corrupt = float(corrupt_probs[target_id].item())

        num_layers = len(self.model.layers)
        indirect_effects = np.zeros((num_layers, seq_len), dtype=np.float32)

        # 3. Intervened Runs: Restore hidden state at (layer, token)
        for layer_idx in range(num_layers):
            clean_layer_hidden = clean_hiddens[layer_idx + 1] # 1-indexed for layer outputs
            for token_pos in range(seq_len):
                # Run forward pass with corrupted input, restoring hidden state at (layer_idx, token_pos)
                curr = corrupted_embed.clone()
                for l_i, layer in enumerate(self.model.layers):
                    curr = layer(curr)
                    if l_i == layer_idx:
                        # Restore clean activation at token_pos
                        curr[:, token_pos, :] = clean_layer_hidden[:, token_pos, :]

                curr = self.model.ln_f(curr)
                intervened_logits = self.model.lm_head(curr)
                intervened_probs = F.softmax(intervened_logits[0, -1, :], dim=-1)
                p_intervened = float(intervened_probs[target_id].item())

                # Normalized Indirect Effect
                denom = max(p_clean - p_corrupt, 1e-4)
                ie = (p_intervened - p_corrupt) / denom
                indirect_effects[layer_idx, token_pos] = float(np.clip(ie, 0.0, 1.0))

        layer_names = [f"layer_{i}" for i in range(num_layers)]
        return CausalTraceResult(
            prompt=prompt,
            subject=subj_str,
            target=target,
            tokens=tokens,
            clean_prob=p_clean,
            corrupt_prob=p_corrupt,
            indirect_effects=indirect_effects,
            layer_names=layer_names,
        )
