#!/usr/bin/env python3
"""Plot performance timings from output.txt as a horizontal bar chart.

Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


TIME_PATTERN = re.compile(r"^\s*time\s*\[(.+?)\]\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$")


def parse_times(path: Path) -> dict[str, float]:
    """Parse timing lines of the form: time [label] = value s."""
    times: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TIME_PATTERN.match(line)
        if match:
            label = match.group(1).strip()
            value = float(match.group(2))
            times[label] = value
    return times


def build_plot_data(
    times: dict[str, float],
) -> tuple[list[str], list[float], list[str], list[str]]:
    """Build ordered plotting vectors.

    Returns (labels, values, colors, y_tick_labels).
    """
    # Explicit plotting order requested by the user.
    order = [
        "evaluation   (isotropic, Jax)",
        "evaluation  (isotropic, CUDA)",
        "evaluation (anisotropic, Jax)",
        "evaluation  (anisotropic, CUDA)",
        "gradient (isotropic)",
        "gradient (anisotropic)",
    ]

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    y_tick_labels: list[str] = []

    pretty_name = {
        "evaluation   (isotropic, Jax)": "Isotropic JAX",
        "evaluation  (isotropic, CUDA)": "Isotropic CUDA",
        "evaluation (anisotropic, Jax)": "Anisotropic JAX",
        "evaluation  (anisotropic, CUDA)": "Anisotropic CUDA",
        "gradient (isotropic)": "Gradient Isotropic",
        "gradient (anisotropic)": "Gradient Anisotropic",
    }

    for key in order:
        if key not in times:
            continue
        labels.append(key)
        values.append(times[key])
        y_tick_labels.append(pretty_name[key])
        if key.startswith("gradient"):
            colors.append("#2ca02c")  # green
        elif "CUDA" in key:
            colors.append("#1f77b4")  # blue
        else:
            colors.append("#d62728")  # red (JAX)

    if not labels:
        raise ValueError("No timing entries were found in the input file.")

    return labels, values, colors, y_tick_labels


def plot_performance(times: dict[str, float], output: Path) -> None:
    """Create horizontal bar chart for performance timings."""
    labels, values, colors, y_tick_labels = build_plot_data(times)

    y_pos = [0.0, 1.0, 2.6, 3.6, 5.2, 6.2][: len(values)]
    fig_height = max(4.0, 0.8 * len(values) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    max_value = max(values)

    bars = ax.barh(y_pos, values, color=colors, edgecolor="none", linewidth=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_tick_labels, fontsize=16)
    ax.invert_yaxis()
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_title("JAX vs CUDA Performance", fontsize=18)
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, 1.28 * max_value)

    for bar, value in zip(bars, values):
        ax.text(
            value + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} s",
            va="center",
            ha="left",
            fontsize=15,
        )

    def add_speedup_arrow(jax_label: str, cuda_label: str) -> None:
        if jax_label not in labels or cuda_label not in labels:
            return

        jax_idx = labels.index(jax_label)
        cuda_idx = labels.index(cuda_label)
        jax_time = values[jax_idx]
        cuda_time = values[cuda_idx]
        y_arrow = y_pos[cuda_idx] - 0.42
        speedup = jax_time / cuda_time
        gain = "speedup" if speedup > 1 else "slowdown"
        ax.text(
            max(cuda_time, jax_time) + 0.05,
            y_arrow + 0.0,
            f"{speedup:.1f}x {gain}",
            ha="left",
            va="center",
            fontsize=18,
            color="#5f5f5f",
            fontweight="bold",
        )

    add_speedup_arrow("evaluation   (isotropic, Jax)", "evaluation  (isotropic, CUDA)")
    add_speedup_arrow(
        "evaluation (anisotropic, Jax)", "evaluation  (anisotropic, CUDA)"
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timing data from output.txt")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("output.txt"),
        help="Path to input timing text file (default: src/output.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("performance.png"),
        help="Output image path (default: src/performance.png).",
    )
    args = parser.parse_args()

    times = parse_times(args.input)
    plot_performance(times, args.output)


if __name__ == "__main__":
    main()
