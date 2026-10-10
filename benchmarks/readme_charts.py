"""Render the charts shown in the README.

The forecast and components charts come straight from the library's default
plot style, so they change whenever ``prophet/plot.py`` does. The throughput
chart reads the JSON written by ``compare_upstream.py`` (run it through
``notebooks/benchmark_vs_upstream.ipynb``), so no numbers are typed in by hand.

Usage::

    uv run python benchmarks/readme_charts.py \
        --examples-dir examples \
        --results-dir <folder with upstream.json and neoprophet_mlx.json> \
        --out-dir .github/assets
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from prophet import forecast  # noqa: E402
from prophet.plot import ACCENT, GRID, INK, INK_SOFT, OBSERVED, PAPER  # noqa: E402

logger = logging.getLogger("readme_charts")

DPI = 150
SANS_STACK = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
SERIF_STACK = ("Georgia", "Charter", "Times New Roman", "DejaVu Serif")
THROUGHPUT_SERIES = 100


@dataclass(frozen=True)
class Bar:
    label: str
    seconds: float
    highlight: bool


@dataclass(frozen=True)
class Throughput:
    n_series: int
    n_rows: int
    bars: tuple[Bar, ...]
    speedup: float
    max_rel_diff: float
    cpu: str


@cache
def installed(stack: tuple[str, ...]) -> list[str]:
    # Filtering avoids matplotlib's findfont warning for every missing family.
    names = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in stack if name in names] or [stack[-1]]


def entry_for(results: Mapping[str, Any], n_series: int) -> Mapping[str, Any]:
    matches = [t for t in results.get("throughput", []) if t["n_series"] == n_series]
    if not matches:
        raise ValueError(f"No throughput entry for {n_series} series in the results.")
    return matches[0]


def throughput_summary(
    upstream: Mapping[str, Any], fork_mlx: Mapping[str, Any], n_series: int
) -> Throughput:
    up = entry_for(upstream, n_series)
    fork = entry_for(fork_mlx, n_series)
    up_s = float(up["cmdstan"]["median_s"])
    mlx_s = float(fork["mlx"]["median_s"])
    bars = (
        Bar(
            f"Upstream Prophet {upstream['metadata']['prophet_version']}, one by one",
            up_s,
            False,
        ),
        Bar(
            "NeoProphet with CmdStan, one by one",
            float(fork["cmdstan"]["median_s"]),
            False,
        ),
        Bar("NeoProphet MLX, fit_many() in one batch", mlx_s, True),
    )
    return Throughput(
        n_series=n_series,
        n_rows=int(fork["n_rows"]),
        bars=bars,
        speedup=up_s / mlx_s,
        max_rel_diff=float(fork["mlx"]["max_rel_yhat_diff"]),
        cpu=str(fork_mlx["metadata"]["cpu"]),
    )


def bar_text(bar: Bar, speedup: float) -> str:
    seconds = f"{bar.seconds:.1f} s"
    return f"{seconds}, {speedup:.0f}x faster" if bar.highlight else seconds


def throughput_figure(t: Throughput) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.68, bottom=0.22)

    positions = np.arange(len(t.bars))[::-1]
    for y, bar in zip(positions, t.bars):
        color = ACCENT if bar.highlight else OBSERVED
        ax.barh(y, bar.seconds, height=0.42, color=color, edgecolor=PAPER, linewidth=2)
        ax.text(
            0,
            y + 0.36,
            bar.label,
            ha="left",
            va="bottom",
            color=INK,
            fontsize=10,
            family=installed(SANS_STACK),
        )
        ax.text(
            bar.seconds,
            y,
            f"  {bar_text(bar, t.speedup)}",
            ha="left",
            va="center",
            color=INK if bar.highlight else INK_SOFT,
            fontsize=10,
            fontweight="bold" if bar.highlight else "normal",
            family=installed(SANS_STACK),
        )

    longest = max(b.seconds for b in t.bars)
    ax.set_xlim(0, longest * 1.22)
    ax.set_ylim(-0.5, len(t.bars) - 0.2)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=9.5, length=0)
    for label in ax.get_xticklabels():
        label.set_family(installed(SANS_STACK))
    ax.set_xlabel(
        "seconds to fit", color=INK_SOFT, fontsize=9.5, family=installed(SANS_STACK)
    )

    fig.text(
        0.04,
        0.9,
        f"Fit {t.n_series} series in {t.bars[-1].seconds:.1f} seconds",
        color=INK,
        fontsize=17,
        family=installed(SERIF_STACK),
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.765,
        f"{t.n_series} noisy copies of the Peyton Manning series ({t.n_rows:,} daily rows each), "
        f"median of 3 runs on an {t.cpu}.\nMLX forecasts stay within "
        f"{t.max_rel_diff:.1%} of the data range of the CmdStan fits.",
        color=INK_SOFT,
        fontsize=10,
        family=installed(SANS_STACK),
        linespacing=1.5,
    )
    fig.text(
        0.04,
        0.03,
        "Source: benchmarks/compare_upstream.py via notebooks/benchmark_vs_upstream.ipynb",
        color=INK_SOFT,
        fontsize=8.5,
        family=installed(SANS_STACK),
    )
    return fig


def showcase_figures(examples: Path) -> dict[str, Figure]:
    # A fixed seed keeps the uncertainty bands, and so the PNGs, reproducible.
    np.random.seed(42)
    peyton = pd.read_csv(examples / "example_wp_log_peyton_manning.csv")
    peyton_result = forecast(peyton, horizon="12 months", country_holidays="US")
    air = pd.read_csv(examples / "example_air_passengers.csv")
    air_result = forecast(air, horizon="24 months", seasonality_mode="multiplicative")
    return {
        "forecast.png": peyton_result.plot(),
        "components.png": air_result.plot_components(),
    }


def save(figures: Mapping[str, Figure], out_dir: Path) -> Sequence[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = out_dir / name
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the charts shown in the README."
    )
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    upstream = json.loads((args.results_dir / "upstream.json").read_text())
    fork_mlx = json.loads((args.results_dir / "neoprophet_mlx.json").read_text())
    summary = throughput_summary(upstream, fork_mlx, THROUGHPUT_SERIES)

    figures = {
        **showcase_figures(args.examples_dir),
        "throughput.png": throughput_figure(summary),
    }
    for path in save(figures, args.out_dir):
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
