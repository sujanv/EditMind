"""Tests for EditMind Delta Compression Engine."""

import pytest
import torch

from editmind.core.compression import (
    CompressedDelta,
    CompressionMethod,
    DeltaCompressor,
)


def test_int8_compression_and_decompression():
    torch.manual_seed(42)
    delta = torch.randn(16, 32)

    comp = DeltaCompressor.compress(delta, method=CompressionMethod.INT8)
    assert isinstance(comp, CompressedDelta)
    assert comp.compression_ratio > 3.0  # float32 (4 bytes) -> int8 (1 byte)
    assert comp.fidelity_cosine_sim > 0.999
    assert comp.relative_l2_error < 0.05

    recon = comp.decompress()
    assert recon.shape == delta.shape
    assert torch.allclose(delta, recon, atol=0.08)


def test_int4_compression_and_decompression():
    torch.manual_seed(42)
    delta = torch.randn(20, 20)

    comp = DeltaCompressor.compress(delta, method=CompressionMethod.INT4)
    assert comp.compression_ratio > 6.0  # float32 (4 bytes) -> int4 (0.5 byte)
    assert comp.fidelity_cosine_sim > 0.95

    recon = comp.decompress()
    assert recon.shape == delta.shape
    assert torch.allclose(delta, recon, atol=0.35)


def test_sparse_topk_compression():
    torch.manual_seed(42)
    # Create sparse-like delta with a few significant updates
    delta = torch.randn(10, 10) * 0.01
    delta[2, 3] = 2.5
    delta[5, 5] = -3.1
    delta[8, 1] = 1.9

    comp = DeltaCompressor.compress(delta, method=CompressionMethod.SPARSE_TOPK, sparsity=0.90)
    recon = comp.decompress()

    assert recon.shape == delta.shape
    assert recon[2, 3].item() == pytest.approx(2.5, abs=1e-4)
    assert recon[5, 5].item() == pytest.approx(-3.1, abs=1e-4)
    assert recon[8, 1].item() == pytest.approx(1.9, abs=1e-4)


def test_hybrid_compression():
    torch.manual_seed(42)
    delta = torch.randn(30, 30)

    comp = DeltaCompressor.compress(delta, method=CompressionMethod.HYBRID, sparsity=0.8)
    recon = comp.decompress()

    assert recon.shape == delta.shape
    assert comp.compressed_bytes < comp.original_bytes
    d = comp.to_dict()
    assert "compression_ratio" in d
    assert "fidelity_cosine_sim" in d


def test_fidelity_edge_cases():
    zeros = torch.zeros(5, 5)
    f = DeltaCompressor.evaluate_fidelity(zeros, zeros)
    assert f["cosine_similarity"] == 1.0
    assert f["relative_l2_error"] == 0.0

    ones = torch.ones(5, 5)
    neg_ones = -torch.ones(5, 5)
    f_neg = DeltaCompressor.evaluate_fidelity(ones, neg_ones)
    assert f_neg["cosine_similarity"] == pytest.approx(-1.0, abs=1e-4)
