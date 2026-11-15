"""EditMind Weight Delta Compression and Low-Bit Quantization Engine.

Provides low-bit symmetric quantization (INT8, INT4), magnitude-aware
top-k sparse pruning, and hybrid sparse-quantized representations for
knowledge editing parameter deltas (Delta W) to drastically minimize
in-memory cache footprints and replication network payloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch


class CompressionMethod(str, Enum):
    """Supported compression strategies for editing parameter deltas."""
    INT8 = "int8"
    INT4 = "int4"
    SPARSE_TOPK = "sparse_topk"
    HYBRID = "hybrid"


@dataclass
class CompressedDelta:
    """Container for compressed parameter delta representations."""
    method: str
    original_shape: Tuple[int, ...]
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    fidelity_cosine_sim: float
    relative_l2_error: float
    data: Dict[str, Any]

    def decompress(self) -> torch.Tensor:
        """Decompress and reconstruct parameter delta tensor."""
        return DeltaCompressor.decompress(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for telemetry and JSON serialization."""
        return {
            "method": self.method,
            "original_shape": list(self.original_shape),
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": round(self.compression_ratio, 2),
            "fidelity_cosine_sim": round(self.fidelity_cosine_sim, 5),
            "relative_l2_error": round(self.relative_l2_error, 5),
        }


class DeltaCompressor:
    """Compressor for neural weight edit deltas."""

    @staticmethod
    def compress(
        delta: torch.Tensor,
        method: CompressionMethod = CompressionMethod.INT8,
        sparsity: float = 0.8,
    ) -> CompressedDelta:
        """Compress parameter delta tensor using selected strategy."""
        if not isinstance(delta, torch.Tensor):
            raise TypeError("delta must be a torch.Tensor")

        orig_shape = delta.shape
        orig_bytes = delta.numel() * delta.element_size()

        if method == CompressionMethod.INT8:
            compressed_data, comp_bytes = DeltaCompressor._compress_int8(delta)
        elif method == CompressionMethod.INT4:
            compressed_data, comp_bytes = DeltaCompressor._compress_int4(delta)
        elif method == CompressionMethod.SPARSE_TOPK:
            compressed_data, comp_bytes = DeltaCompressor._compress_sparse(delta, sparsity=sparsity)
        elif method == CompressionMethod.HYBRID:
            compressed_data, comp_bytes = DeltaCompressor._compress_hybrid(delta, sparsity=sparsity)
        else:
            raise ValueError(f"Unknown compression method: {method}")

        wrapper = CompressedDelta(
            method=method.value if isinstance(method, CompressionMethod) else str(method),
            original_shape=orig_shape,
            original_bytes=orig_bytes,
            compressed_bytes=comp_bytes,
            compression_ratio=orig_bytes / max(comp_bytes, 1),
            fidelity_cosine_sim=1.0,
            relative_l2_error=0.0,
            data=compressed_data,
        )

        # Compute reconstruction fidelity
        recon = DeltaCompressor.decompress(wrapper)
        fidelity = DeltaCompressor.evaluate_fidelity(delta, recon)
        wrapper.fidelity_cosine_sim = fidelity["cosine_similarity"]
        wrapper.relative_l2_error = fidelity["relative_l2_error"]

        return wrapper

    @staticmethod
    def decompress(compressed: CompressedDelta) -> torch.Tensor:
        """Reconstruct float tensor from CompressedDelta container."""
        method = compressed.method
        data = compressed.data
        shape = compressed.original_shape

        if method == CompressionMethod.INT8.value:
            quant = data["quantized"].to(torch.float32)
            scale = data["scale"]
            return (quant * scale).view(shape)

        elif method == CompressionMethod.INT4.value:
            packed = data["packed"]
            scale = data["scale"]
            total_elements = math.prod(shape)
            # Unpack 2 4-bit nibbles per int8 byte
            high = (packed >> 4) & 0x0F
            low = packed & 0x0F
            # Convert unsigned 4-bit (0..15) back to signed (-8..7)
            high = torch.where(high >= 8, high - 16, high).to(torch.float32)
            low = torch.where(low >= 8, low - 16, low).to(torch.float32)
            interleaved = torch.stack([high, low], dim=-1).flatten()[:total_elements]
            return (interleaved * scale).view(shape)

        elif method == CompressionMethod.SPARSE_TOPK.value:
            indices = data["indices"]
            values = data["values"]
            total_elements = math.prod(shape)
            flat = torch.zeros(total_elements, dtype=values.dtype)
            if indices.numel() > 0:
                flat[indices] = values
            return flat.view(shape)

        elif method == CompressionMethod.HYBRID.value:
            indices = data["indices"]
            quant = data["quant_values"].to(torch.float32)
            scale = data["scale"]
            total_elements = math.prod(shape)
            flat = torch.zeros(total_elements, dtype=torch.float32)
            if indices.numel() > 0:
                flat[indices] = quant * scale
            return flat.view(shape)

        raise ValueError(f"Unsupported decompress method: {method}")

    @staticmethod
    def _compress_int8(tensor: torch.Tensor) -> Tuple[Dict[str, Any], int]:
        """Symmetric INT8 quantization."""
        flat = tensor.flatten().to(torch.float32)
        max_abs = flat.abs().max().item()
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1e-12

        quant = torch.clamp(torch.round(flat / scale), -128, 127).to(torch.int8)
        bytes_count = quant.numel() * 1 + 4  # int8 bytes + float32 scale
        return {"quantized": quant, "scale": scale}, bytes_count

    @staticmethod
    def _compress_int4(tensor: torch.Tensor) -> Tuple[Dict[str, Any], int]:
        """Symmetric INT4 quantization packed into INT8."""
        flat = tensor.flatten().to(torch.float32)
        max_abs = flat.abs().max().item()
        scale = max_abs / 7.0 if max_abs > 1e-12 else 1e-12

        # Clamp into signed 4-bit [-8, 7]
        quant = torch.clamp(torch.round(flat / scale), -8, 7).to(torch.int8)
        # Pad if odd
        if quant.numel() % 2 != 0:
            quant = torch.cat([quant, torch.zeros(1, dtype=torch.int8)])

        # Convert signed to 4-bit unsigned (mask with 0x0F)
        u4 = quant & 0x0F
        high = u4[0::2]
        low = u4[1::2]
        packed = ((high << 4) | low).to(torch.int8)
        bytes_count = packed.numel() * 1 + 4  # packed bytes + float32 scale
        return {"packed": packed, "scale": scale}, bytes_count

    @staticmethod
    def _compress_sparse(tensor: torch.Tensor, sparsity: float) -> Tuple[Dict[str, Any], int]:
        """Magnitude-based Top-K sparse pruning."""
        flat = tensor.flatten().to(torch.float32)
        total = flat.numel()
        k = max(1, int(round((1.0 - sparsity) * total)))

        abs_vals = flat.abs()
        topk_vals, topk_indices = torch.topk(abs_vals, k=k, largest=True, sorted=False)
        stored_vals = flat[topk_indices]

        # int64 indices (8 bytes) + float32 values (4 bytes)
        bytes_count = topk_indices.numel() * 4 + stored_vals.numel() * 4
        return {"indices": topk_indices.to(torch.int32), "values": stored_vals}, bytes_count

    @staticmethod
    def _compress_hybrid(tensor: torch.Tensor, sparsity: float) -> Tuple[Dict[str, Any], int]:
        """Hybrid top-k sparsity combined with INT8 quantization."""
        flat = tensor.flatten().to(torch.float32)
        total = flat.numel()
        k = max(1, int(round((1.0 - sparsity) * total)))

        abs_vals = flat.abs()
        _, topk_indices = torch.topk(abs_vals, k=k, largest=True, sorted=False)
        stored_vals = flat[topk_indices]

        max_abs = stored_vals.abs().max().item() if stored_vals.numel() > 0 else 1.0
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1e-12
        quant = torch.clamp(torch.round(stored_vals / scale), -128, 127).to(torch.int8)

        # int32 indices (4 bytes) + int8 values (1 byte) + float32 scale (4 bytes)
        bytes_count = topk_indices.numel() * 4 + quant.numel() * 1 + 4
        return {
            "indices": topk_indices.to(torch.int32),
            "quant_values": quant,
            "scale": scale,
        }, bytes_count

    @staticmethod
    def evaluate_fidelity(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute mathematical fidelity between original and reconstructed tensors."""
        flat_orig = original.flatten().to(torch.float32)
        flat_recon = reconstructed.flatten().to(torch.float32)

        norm_orig = torch.norm(flat_orig)
        norm_recon = torch.norm(flat_recon)

        if norm_orig < 1e-12 or norm_recon < 1e-12:
            cosine_sim = 1.0 if norm_orig < 1e-12 and norm_recon < 1e-12 else 0.0
        else:
            cosine_sim = float(torch.dot(flat_orig, flat_recon) / (norm_orig * norm_recon))

        diff = flat_orig - flat_recon
        diff_norm = torch.norm(diff)
        rel_l2 = float(diff_norm / (norm_orig + 1e-12))
        max_abs = float(diff.abs().max().item())

        return {
            "cosine_similarity": float(cosine_sim),
            "relative_l2_error": float(rel_l2),
            "max_absolute_error": float(max_abs),
        }
