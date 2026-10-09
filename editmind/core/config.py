"""Configuration management for EditMind."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml


@dataclass
class ModelConfig:
    name: str = "toy_causal_lm"
    vocab_size: int = 1000
    hidden_dim: int = 128
    num_layers: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    max_seq_len: int = 128
    device: str = "cpu"


@dataclass
class ROMEConfig:
    method: str = "rome"
    target_layer: int = 3
    v_num_grad_steps: int = 25
    v_lr: float = 0.1
    v_loss_layer: int = 5
    v_weight_decay: float = 0.1
    clamp_norm_factor: float = 4.0
    rewrite_module_tmp: str = "layers.{}.mlp.down_proj"
    layer_norm_module_tmp: str = "layers.{}.ln2"


@dataclass
class MEMITConfig:
    method: str = "memit"
    layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    v_num_grad_steps: int = 20
    v_lr: float = 0.1
    v_weight_decay: float = 0.1
    clamp_norm_factor: float = 4.0
    rewrite_module_tmp: str = "layers.{}.mlp.down_proj"


@dataclass
class MENDConfig:
    method: str = "mend"
    target_layers: List[int] = field(default_factory=lambda: [3, 4])
    lr: float = 1e-3
    hypernet_hidden_dim: int = 64
    rank: int = 1
    grad_clip: float = 1.0
    combine: bool = True
    norm: bool = True
    rewrite_module_tmp: str = "layers.{}.mlp.down_proj"


@dataclass
class GRACEConfig:
    method: str = "grace"
    target_layer: int = 3
    epsilon: float = 1.5
    metric: str = "euclidean"
    inner_lr: float = 0.05
    inner_steps: int = 10
    replacement: str = "replace"
    codebook_size_limit: int = 1000


@dataclass
class IKEConfig:
    method: str = "ike"
    k_demonstrations: int = 3
    retrieval_strategy: str = "similarity"
    use_analogy_demos: bool = True
    use_copy_demos: bool = True


@dataclass
class FTConfig:
    method: str = "ft_l"
    target_layer: int = 3
    lr: float = 5e-4
    num_steps: int = 25
    norm_constraint: str = "l_inf"
    max_norm: float = 0.1
    kl_factor: float = 0.05
    weight_decay: float = 0.01


@dataclass
class EditMindConfig:
    project_name: str = "EditMind"
    device: str = "cpu"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EditMindConfig:
        model_data = data.get("model", {})
        model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in ModelConfig.__dataclass_fields__})
        return cls(
            project_name=data.get("project", {}).get("name", "EditMind"),
            device=data.get("device", "cpu"),
            seed=data.get("seed", 42),
            model=model_cfg,
            raw_config=data,
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file from path."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
