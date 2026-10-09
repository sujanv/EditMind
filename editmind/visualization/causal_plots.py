"""Visualizers for Causal Mediation Analysis and Activation Traces."""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

from editmind.models.causal_tracer import CausalTraceResult


def render_ascii_causal_heatmap(trace_result: CausalTraceResult) -> str:
    """Renders a terminal-friendly ASCII heatmap of indirect effects across layers and tokens."""
    matrix = trace_result.indirect_effects
    num_layers, seq_len = matrix.shape
    tokens = trace_result.tokens

    # Block shades for intensity
    shades = ["  ", "░░", "▒▒", "▓▓", "██"]

    lines = []
    lines.append(f"\n=== CAUSAL MEDIATION HEATMAP ===")
    lines.append(f"Prompt : '{trace_result.prompt}'")
    lines.append(f"Subject: '{trace_result.subject}' | Target: '{trace_result.target}'")
    lines.append(f"P(Clean): {trace_result.clean_prob:.4f} | P(Corrupt): {trace_result.corrupt_prob:.4f}\n")

    # Header with tokens
    token_headers = [f"{t[:6]:^6}" for t in tokens]
    lines.append(f"{'Layer':<8} | " + " ".join(token_headers))
    lines.append("-" * (11 + 7 * seq_len))

    for l_idx in range(num_layers - 1, -1, -1):
        row_str = f"L{l_idx:<6} | "
        for t_idx in range(seq_len):
            val = matrix[l_idx, t_idx]
            shade_idx = min(int(val * len(shades)), len(shades) - 1)
            row_str += f"  {shades[shade_idx]}  "
        lines.append(row_str)

    best_l, best_t, tok = trace_result.find_critical_layer_and_token()
    lines.append("-" * (11 + 7 * seq_len))
    lines.append(f"Peak Mediating Site: Layer {best_l}, Token '{tok}' (AIE = {matrix[best_l, best_t]:.3f})\n")
    return "\n".join(lines)


def generate_svg_causal_heatmap(trace_result: CausalTraceResult, cell_size: int = 40) -> str:
    """Generates an SVG heatmap string suitable for rendering in web dashboards."""
    matrix = trace_result.indirect_effects
    num_layers, seq_len = matrix.shape
    tokens = trace_result.tokens

    margin_left = 80
    margin_top = 50
    width = margin_left + seq_len * cell_size + 40
    height = margin_top + num_layers * cell_size + 60

    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; font-family:monospace; border-radius:8px;">',
        f'<text x="20" y="30" fill="#cdd6f4" font-size="14" font-weight="bold">Causal Mediation Trace: {trace_result.subject} &rarr; {trace_result.target}</text>',
    ]

    # Column labels (Tokens)
    for t_idx, token in enumerate(tokens):
        x = margin_left + t_idx * cell_size + cell_size // 2
        y = margin_top - 10
        svg_parts.append(
            f'<text x="{x}" y="{y}" fill="#a6adc8" font-size="11" text-anchor="middle">{token[:7]}</text>'
        )

    # Heatmap cells
    for l_idx in range(num_layers):
        # Draw from bottom to top
        display_layer = num_layers - 1 - l_idx
        y = margin_top + l_idx * cell_size
        svg_parts.append(
            f'<text x="{margin_left - 15}" y="{y + cell_size // 2 + 4}" fill="#a6adc8" font-size="11" text-anchor="end">L{display_layer}</text>'
        )

        for t_idx in range(seq_len):
            x = margin_left + t_idx * cell_size
            val = float(matrix[display_layer, t_idx])
            # Color scale: interpolation from dark purple (#313244) to bright teal/cyan (#89dceb) to gold (#f9e2af)
            r = int(49 + val * 200)
            g = int(50 + val * 170)
            b = int(68 + (1.0 - val) * 100)
            color = f"rgb({r},{g},{b})"
            svg_parts.append(
                f'<rect x="{x+1}" y="{y+1}" width="{cell_size-2}" height="{cell_size-2}" rx="4" fill="{color}">'
                f'<title>L{display_layer}, Token {tokens[t_idx]}: {val:.3f}</title></rect>'
            )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)
