"""Unified model wrappers for PyTorch Causal Language Models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.core.types import ModelOutput


class SimpleTokenizer:
    """Lightweight self-contained word & subword tokenizer."""

    def __init__(self, initial_vocab: Optional[List[str]] = None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        base_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        default_words = [
            "The", "the", "Eiffel", "Tower", "is", "in", "Paris", "Rome", "London",
            "France", "Italy", "UK", "capital", "of", "Einstein", "discovered",
            "relativity", "quantum", "physics", "Messi", "plays", "for", "PSG",
            "Barcelona", "Miami", "Apple", "CEO", "Cook", "Jobs", "Tesla",
            "founded", "by", "Musk", "Earth", "orbits", "Sun", "Moon",
            "Jupiter", "planet", "solar", "system", "Berlin", "Germany",
            "Madrid", "Spain", "Tokyo", "Japan", "created", "invented",
            "located", "born", "works", "at", "author", "wrote", "Hamlet",
            "Shakespeare", "Turing", "father", "computing", "Python",
            "Guido", "van", "Rossum", "Golang", "Ken", "Thompson",
        ]
        tokens = base_tokens + default_words + (initial_vocab or [])
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        for token in tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def encode(self, text: str) -> List[int]:
        words = text.strip().split()
        ids = []
        for word in words:
            clean = word.strip(".,;:?!'\"()")
            if clean in self.token2id:
                ids.append(self.token2id[clean])
            else:
                idx = self._add_token(clean)
                ids.append(idx)
        return ids or [self.token2id[self.unk_token]]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.id2token.get(idx, self.unk_token) for idx in token_ids]
        return " ".join(t for t in tokens if t not in [self.pad_token, self.bos_token, self.eos_token])

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.c_proj(context)


class TransformerMLP(nn.Module):
    """Feedforward associative memory block."""
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        intermediate_dim = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = TransformerMLP(hidden_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyCausalLM(nn.Module):
    """Lightweight real PyTorch causal transformer model for fast knowledge editing experiments."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.wpe = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        hidden = self.wte(input_ids) + self.wpe(pos)
        hidden_states = [hidden] if output_hidden_states else None

        for layer in self.layers:
            hidden = layer(hidden)
            if output_hidden_states:
                hidden_states.append(hidden)

        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        return logits, hidden_states


class BaseModelWrapper(ABC):
    """Abstract interface for model execution and parameter manipulation."""

    @abstractmethod
    def predict_next_token_probs(self, prompt: str) -> Dict[str, float]:
        pass

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        pass

    @abstractmethod
    def get_module(self, path: str) -> nn.Module:
        pass

    @abstractmethod
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        pass


class UnifiedModelWrapper(BaseModelWrapper):
    """Unified wrapper around PyTorch causal language models."""

    def __init__(self, model: nn.Module, tokenizer: Optional[SimpleTokenizer] = None, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.num_layers = getattr(model, "num_layers", len(getattr(model, "layers", [])))

    @classmethod
    def create_toy_model(cls, hidden_dim: int = 128, num_layers: int = 6) -> UnifiedModelWrapper:
        tokenizer = SimpleTokenizer()
        model = ToyCausalLM(
            vocab_size=max(1000, tokenizer.vocab_size + 100),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4,
        )
        return cls(model=model, tokenizer=tokenizer, device="cpu")

    def tokenize(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False) -> ModelOutput:
        logits, hidden_states = self.model(input_ids, output_hidden_states=output_hidden_states)
        return ModelOutput(
            logits=logits,
            hidden_states=hidden_states,
            token_ids=input_ids[0].tolist(),
        )

    def predict_next_token_probs(self, prompt: str) -> Dict[str, float]:
        """Calculates softmax probabilities for next token candidates."""
        self.model.eval()
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

        result: Dict[str, float] = {}
        for token_str, token_id in self.tokenizer.token2id.items():
            if token_id < probs.shape[0]:
                result[token_str] = float(probs[token_id].item())
        return result

    def get_module(self, path: str) -> nn.Module:
        """Resolves module path like 'layers.3.mlp.down_proj'."""
        curr = self.model
        for part in path.split("."):
            if part.isdigit():
                curr = curr[int(part)]
            else:
                curr = getattr(curr, part)
        return curr

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state)

    def generate(self, prompt: str, max_new_tokens: int = 5) -> str:
        """Autoregressive text generation."""
        self.model.eval()
        curr_ids = self.tokenize(prompt)
        generated_ids: List[int] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.model(curr_ids)
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                generated_ids.append(next_id)
                curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]], device=self.device)], dim=1)

        return self.detokenize(generated_ids)
