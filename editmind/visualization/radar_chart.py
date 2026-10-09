"""Radar chart generator for multi-method knowledge editing comparison."""

from __future__ import annotations
from typing import Dict, List
import math

from editmind.core.types import EvaluationMetrics


def generate_svg_radar_chart(
    method_metrics: Dict[str, EvaluationMetrics],
    size: int = 400,
) -> str:
    """Generates an SVG radar chart comparing multiple methods across 5 core dimensions."""
    axes = ["Efficacy", "Generality", "Locality", "Portability", "Retention"]
    num_axes = len(axes)
    cx, cy = size / 2, size / 2
    radius = size * 0.36

    svg_parts = [
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; font-family:sans-serif; border-radius:8px;">',
        f'<text x="{cx}" y="28" fill="#cdd6f4" font-size="14" font-weight="bold" text-anchor="middle">Comparative Knowledge Editing Radar</text>',
    ]

    # Draw grid rings
    for ring in [0.2, 0.4, 0.6, 0.8, 1.0]:
        r = radius * ring
        ring_points = []
        for i in range(num_axes):
            angle = (2 * math.pi / num_axes) * i - math.pi / 2
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            ring_points.append(f"{x:.1f},{y:.1f}")
        svg_parts.append(f'<polygon points="{" ".join(ring_points)}" fill="none" stroke="#45475a" stroke-width="1" stroke-dasharray="2,2"/>')

    # Draw axis lines & labels
    for i, axis_name in enumerate(axes):
        angle = (2 * math.pi / num_axes) * i - math.pi / 2
        x_end = cx + radius * math.cos(angle)
        y_end = cy + radius * math.sin(angle)
        svg_parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x_end:.1f}" y2="{y_end:.1f}" stroke="#585b70" stroke-width="1"/>')

        # Label positioning
        lx = cx + (radius + 24) * math.cos(angle)
        ly = cy + (radius + 24) * math.sin(angle)
        svg_parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#a6adc8" font-size="11" text-anchor="middle" dominant-baseline="middle">{axis_name}</text>')

    # Palette for methods
    colors = {
        "rome": "#89b4fa",       # Blue
        "memit": "#a6e3a1",      # Green
        "mend": "#f9e2af",       # Yellow
        "grace": "#cba6f7",      # Mauve
        "ike": "#f38ba8",        # Red
        "ft_l": "#fab387",       # Peach
        "lora_edit": "#94e2d5",  # Teal
    }

    # Draw polygons for each method
    legend_y = size - 20
    for idx, (method_name, m) in enumerate(method_metrics.items()):
        color = colors.get(method_name.lower(), "#89b4fa")
        values = [
            m.efficacy,
            m.generality,
            m.locality,
            m.portability,
            m.ripple_retention,
        ]
        poly_points = []
        for i, val in enumerate(values):
            angle = (2 * math.pi / num_axes) * i - math.pi / 2
            val_clamped = max(0.05, min(1.0, val))
            x = cx + radius * val_clamped * math.cos(angle)
            y = cy + radius * val_clamped * math.sin(angle)
            poly_points.append(f"{x:.1f},{y:.1f}")

        svg_parts.append(
            f'<polygon points="{" ".join(poly_points)}" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2"/>'
        )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)
