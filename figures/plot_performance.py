#!/usr/bin/env python3
"""Plot performance timings as horizontal bar charts.

Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt


TIME_PATTERN = re.compile(r"^\s*time\s*\[(.+?)\]\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$")
SECONDS_TO_MILLISECONDS: Final[float] = 1000.0
PLOT_ITEMS: Final[list[dict[str, str | float]]] = [
    {
        "raw_label": "evaluation   (isotropic, double precision)",
        "display_label": "Isotropic: JAX [double]",
        "group": "solve",
        "solver": "jax",
        "y": 0.0,
    },
    {
        "raw_label": "evaluation   (isotropic, single precision)",
        "display_label": "JAX [single]",
        "group": "solve",
        "solver": "jax",
        "y": 1.0,
    },
    {
        "raw_label": "evaluation  (isotropic, CUDA)",
        "display_label": "CUDA [single]",
        "group": "solve",
        "solver": "cuda",
        "y": 2.0,
    },
    {
        "raw_label": "evaluation (anisotropic, double precision)",
        "display_label": "Anisotropic: JAX [double]",
        "group": "solve",
        "solver": "jax",
        "y": 4.0,
    },
    {
        "raw_label": "evaluation (anisotropic, single precision)",
        "display_label": "JAX [single]",
        "group": "solve",
        "solver": "jax",
        "y": 5.0,
    },
    {
        "raw_label": "evaluation  (anisotropic, CUDA)",
        "display_label": "CUDA [single]",
        "group": "solve",
        "solver": "cuda",
        "y": 6.0,
    },
    {
        "raw_label": "gradient (isotropic, double precision)",
        "display_label": "Isotropic: JAX [double]",
        "group": "gradient",
        "solver": "jax",
        "y": 8.0,
    },
    {
        "raw_label": "gradient (isotropic, single precision)",
        "display_label": "JAX [single]",
        "group": "gradient",
        "solver": "jax",
        "y": 9.0,
    },
    {
        "raw_label": "gradient (anisotropic, double precision)",
        "display_label": "Anisotropic: JAX [double]",
        "group": "gradient",
        "solver": "jax",
        "y": 11.0,
    },
    {
        "raw_label": "gradient (anisotropic, single precision)",
        "display_label": "JAX [single]",
        "group": "gradient",
        "solver": "jax",
        "y": 12.0,
    },
]
COLOR_BY_KEY: Final[dict[tuple[str, str], str]] = {
    ("solve", "jax-single"): "#d62728",
    ("solve", "jax-double"): "#ff9896",
    ("solve", "cuda-single"): "#1f77b4",
    ("gradient", "jax-single"): "#2ca02c",
    ("gradient", "jax-double"): "#98df8a",
}


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
) -> tuple[list[float], list[float], list[str], list[str], list[str]]:
    """Build ordered plotting vectors.

    Returns (y_positions, values, colors, y_tick_labels, raw_labels).
    """
    y_positions: list[float] = []
    values: list[float] = []
    colors: list[str] = []
    y_tick_labels: list[str] = []
    raw_labels: list[str] = []

    for item in PLOT_ITEMS:
        raw_label = str(item["raw_label"])
        if raw_label not in times:
            continue
        y_positions.append(float(item["y"]))
        values.append(times[raw_label])
        y_tick_labels.append(str(item["display_label"]))
        raw_labels.append(raw_label)
        precision_key = (
            "single" if "single" in raw_label or "CUDA" in raw_label else "double"
        )
        color_key = (str(item["group"]), f"{item['solver']}-{precision_key}")
        colors.append(COLOR_BY_KEY[color_key])

    if not y_positions:
        raise ValueError("No timing entries were found in the input file.")

    return y_positions, values, colors, y_tick_labels, raw_labels


def plot_performance(times: dict[str, float], output: Path, x_limit: float) -> None:
    """Create horizontal bar chart for performance timings."""
    y_pos, values, colors, y_tick_labels, raw_labels = build_plot_data(times)
    values_ms = [value * SECONDS_TO_MILLISECONDS for value in values]

    fig_height = max(4.0, 0.8 * len(values) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))

    bars = ax.barh(y_pos, values_ms, color=colors, edgecolor="none", linewidth=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_tick_labels, fontsize=14)
    for tick_label in ax.get_yticklabels():
        tick_label.set_ha("right")
        tick_label.set_multialignment("right")
    ax.invert_yaxis()
    ax.set_xlabel("Time [ms]", fontsize=16)
    ax.set_title("JAX vs CUDA Performance", fontsize=18)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", pad=6)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, x_limit)
    ax.axhline(7.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        0.9,
        0.7,
        "forward solve",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=24,
        color="#5f5f5f",
        fontweight="bold",
    )
    ax.text(
        0.9,
        0.27,
        "gradient",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=24,
        color="#5f5f5f",
        fontweight="bold",
    )

    for bar, value_ms in zip(bars, values_ms):
        ax.text(
            value_ms + x_limit * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value_ms:.0f} ms",
            va="center",
            ha="left",
            fontsize=14,
        )

    def add_speedup_annotation(jax_label: str, cuda_label: str) -> None:
        if jax_label not in raw_labels or cuda_label not in raw_labels:
            return

        jax_idx = raw_labels.index(jax_label)
        cuda_idx = raw_labels.index(cuda_label)
        jax_time = values_ms[jax_idx]
        cuda_time = values_ms[cuda_idx]
        speedup = jax_time / cuda_time
        ax.text(
            max(cuda_time, jax_time) + x_limit * 0.1,
            (y_pos[jax_idx] + y_pos[cuda_idx]) / 2,
            f"{speedup:.1f}x speedup",
            ha="left",
            va="center",
            fontsize=18,
            color="#5f5f5f",
            fontweight="bold",
        )

    add_speedup_annotation(
        "evaluation   (isotropic, single precision)",
        "evaluation  (isotropic, CUDA)",
    )
    add_speedup_annotation(
        "evaluation (anisotropic, single precision)",
        "evaluation  (anisotropic, CUDA)",
    )

    fig.subplots_adjust(left=0.36, right=0.98, top=0.92, bottom=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved plot to {output}")


def determine_x_limit(times: dict[str, float]) -> float:
    """Choose the horizontal range for the generated plot."""
    plotted_values = [
        times[str(item["raw_label"])]
        for item in PLOT_ITEMS
        if str(item["raw_label"]) in times
    ]
    if not plotted_values:
        raise ValueError("No recognised timing entries were found in the input file.")
    return 1.28 * max(plotted_values) * SECONDS_TO_MILLISECONDS


def default_input_path() -> Path:
    """Prefer the combined timing output when it is available."""
    return Path(__file__).with_name("output.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot timing data from benchmark output"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Path to input timing text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("performance.png"),
        help="Output image path for the performance plot.",
    )
    args = parser.parse_args()

    times = parse_times(args.input)
    x_limit = determine_x_limit(times)
    plot_performance(times, args.output, x_limit)


if __name__ == "__main__":
    main()
