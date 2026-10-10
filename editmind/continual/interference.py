"""Visualization tools for sequential edit interference matrices."""

from __future__ import annotations
import numpy as np
from editmind.continual.engine import ContinualEditTrajectory


def render_ascii_interference_matrix(trajectory: ContinualEditTrajectory) -> str:
    """Renders ASCII matrix showing mutual interference across sequential edits."""
    mat = trajectory.interference_matrix
    T = mat.shape[0]

    lines = []
    lines.append(f"\n=== CONTINUAL EDIT INTERFERENCE MATRIX (T={T}) ===")
    lines.append(f"Average Retention: {trajectory.average_retention * 100:.1f}%")
    lines.append(f"Catastrophic Forgetting Rate: {trajectory.catastrophic_forgetting_rate * 100:.1f}%\n")

    header = "Fact \\ Step | " + " ".join([f"S{j:<4}" for j in range(T)])
    lines.append(header)
    lines.append("-" * len(header))

    for i in range(T):
        subj = trajectory.requests[i].subject or f"Fact_{i}"
        row = f"{subj[:10]:<10} | "
        for j in range(T):
            if j < i:
                row += " ---  "
            else:
                val = mat[i, j]
                row += f"{val:4.2f} "
        lines.append(row)

    lines.append("-" * len(header))
    return "\n".join(lines)


def generate_svg_interference_matrix(trajectory: ContinualEditTrajectory, cell_size: int = 42) -> str:
    """Generates SVG heatmap of continual edit interference matrix."""
    mat = trajectory.interference_matrix
    T = mat.shape[0]

    margin_left = 120
    margin_top = 60
    width = margin_left + T * cell_size + 40
    height = margin_top + T * cell_size + 60

    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; font-family:monospace; border-radius:8px;">',
        f'<text x="20" y="32" fill="#cdd6f4" font-size="14" font-weight="bold">Continual Edit Retention Matrix</text>',
    ]

    # Step column headers
    for j in range(T):
        x = margin_left + j * cell_size + cell_size // 2
        y = margin_top - 10
        svg_parts.append(f'<text x="{x}" y="{y}" fill="#a6adc8" font-size="11" text-anchor="middle">Step {j}</text>')

    for i in range(T):
        y = margin_top + i * cell_size
        subj = trajectory.requests[i].subject or f"Fact {i}"
        svg_parts.append(f'<text x="{margin_left - 15}" y="{y + cell_size // 2 + 4}" fill="#a6adc8" font-size="11" text-anchor="end">{subj[:12]}</text>')

        for j in range(T):
            x = margin_left + j * cell_size
            if j < i:
                color = "#313244" # Unedited prior to step
                text_val = "-"
            else:
                val = float(mat[i, j])
                # Green for high retention (1.0), red/purple for degraded (0.0)
                r = int(243 * (1.0 - val) + 166 * val)
                g = int(139 * (1.0 - val) + 227 * val)
                b = int(168 * (1.0 - val) + 161 * val)
                color = f"rgb({r},{g},{b})"
                text_val = f"{val:.2f}"

            svg_parts.append(
                f'<rect x="{x+1}" y="{y+1}" width="{cell_size-2}" height="{cell_size-2}" rx="4" fill="{color}"/>'
            )
            svg_parts.append(
                f'<text x="{x + cell_size//2}" y="{y + cell_size//2 + 4}" fill="#11111b" font-size="10" font-weight="bold" text-anchor="middle">{text_val}</text>'
            )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)
