"""Every figure, drawn from results/summary.csv so each plotted number traces to a file.

Figures are built with matplotlib's object API, not pyplot, so no global backend or figure
state is involved and the module works on a machine without a display.
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from decode_sol.analyze import PointSummary, load_point_summaries

logger = logging.getLogger(__name__)

# Light-mode reference palette from the dataviz skill. The first three categorical slots pass
# the colorblind checks for every pair, so they can share a scatter plot.
SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
COLOR_BY_MODEL = {
    "Qwen/Qwen3-1.7B": "#2a78d6",
    "Qwen/Qwen3-4B": "#eb6834",
    "Qwen/Qwen3-8B": "#1baf7a",
}
EFFICIENCY_COLORMAP = LinearSegmentedColormap.from_list(
    "efficiency_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
EFFICIENCY_COLOR_RANGE = (0.6, 0.95)
DARK_CELL_EFFICIENCY = 0.82  # Cells above this are dark enough to need white text.
KV_CACHE_DTYPES = ("auto", "fp8")
LABEL_BY_KV_CACHE_DTYPE = {"auto": "bf16 KV cache", "fp8": "fp8 KV cache"}
LineStyle = Literal["-", "--", "none"]
LINESTYLE_BY_KV_CACHE_DTYPE: dict[str, LineStyle] = {"auto": "-", "fp8": "--"}
NO_LINE_BY_KV_CACHE_DTYPE: dict[str, LineStyle] = {"auto": "none", "fp8": "none"}
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
CONTEXT_LENS = (512, 2048, 8192, 32768)
FULL_NUM_REPEATS = 3
DPI = 200


def to_model_label(model: str) -> str:
    return model.removeprefix("Qwen/")


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    axes.grid(color=GRID_COLOR, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED_TEXT_COLOR, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(GRID_COLOR)


def draw_dtype_marker(axes: Axes, x: float, y: float, *, model: str, kv_cache_dtype: str) -> None:
    """Filled markers for bf16 KV cache, hollow for fp8, so dtype never relies on color."""
    color = COLOR_BY_MODEL[model]
    face_color = color if kv_cache_dtype == "auto" else SURFACE_COLOR
    axes.plot(x, y, "o", markersize=6, color=color, markerfacecolor=face_color, markeredgewidth=1.5)


def build_model_and_dtype_legend_handles(
    *, model_linestyle: LineStyle, linestyle_by_kv_cache_dtype: dict[str, LineStyle]
) -> list[Line2D]:
    handles: list[Line2D] = []
    for model, color in COLOR_BY_MODEL.items():
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle=model_linestyle,
                label=to_model_label(model),
            )
        )
    for kv_cache_dtype in KV_CACHE_DTYPES:
        is_bf16 = kv_cache_dtype == "auto"
        handles.append(
            Line2D(
                [],
                [],
                color=MUTED_TEXT_COLOR,
                marker="o",
                markerfacecolor=MUTED_TEXT_COLOR if is_bf16 else SURFACE_COLOR,
                linestyle=linestyle_by_kv_cache_dtype[kv_cache_dtype],
                label=LABEL_BY_KV_CACHE_DTYPE[kv_cache_dtype],
            )
        )
    return handles


def plot_measured_vs_predicted_tokens_per_s(summaries: list[PointSummary]) -> Figure:
    figure = Figure(figsize=(6.5, 5.5), facecolor=SURFACE_COLOR)
    axes = figure.subplots()
    style_axes(axes)
    for summary in summaries:
        draw_dtype_marker(
            axes,
            summary.predicted_tokens_per_s,
            summary.tokens_per_s,
            model=summary.model,
            kv_cache_dtype=summary.kv_cache_dtype,
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    low, high = axes.get_xlim()
    axes.plot([low, high], [low, high], color=TEXT_COLOR, linewidth=1)
    # Above the line at its low end, where no measurement can sit.
    axes.text(
        low * 1.15,
        low * 1.9,
        "ceiling (measured = predicted)",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TEXT_COLOR,
    )
    axes.set_xlabel("Predicted tokens/s at 265.1 GB/s achievable bandwidth", color=TEXT_COLOR)
    axes.set_ylabel("Measured decode tokens/s", color=TEXT_COLOR)
    efficiencies = [summary.efficiency for summary in summaries]
    axes.set_title(
        f"vLLM decode reaches {min(efficiencies):.0%}-{max(efficiencies):.0%} of the "
        "bandwidth ceiling",
        color=TEXT_COLOR,
        loc="left",
        fontsize=11,
    )
    axes.legend(
        handles=build_model_and_dtype_legend_handles(
            model_linestyle="none", linestyle_by_kv_cache_dtype=NO_LINE_BY_KV_CACHE_DTYPE
        ),
        fontsize=8,
        frameon=False,
        loc="upper left",
    )
    figure.tight_layout()
    return figure


def draw_efficiency_heatmap(axes: Axes, summaries: list[PointSummary]) -> None:
    """Batch on x, context on y. Blank cells did not fit in the KV cache."""
    efficiency_grid = np.full((len(CONTEXT_LENS), len(BATCH_SIZES)), np.nan)
    for summary in summaries:
        row_idx = CONTEXT_LENS.index(summary.context_len)
        column_idx = BATCH_SIZES.index(summary.batch_size)
        efficiency_grid[row_idx, column_idx] = summary.efficiency
        has_full_repeats = summary.num_repeats >= FULL_NUM_REPEATS
        label = f"{summary.efficiency:.2f}" if has_full_repeats else f"{summary.efficiency:.2f}*"
        is_dark_cell = summary.efficiency > DARK_CELL_EFFICIENCY
        axes.text(
            column_idx,
            row_idx,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="white" if is_dark_cell else TEXT_COLOR,
        )
    axes.imshow(
        efficiency_grid,
        cmap=EFFICIENCY_COLORMAP,
        vmin=EFFICIENCY_COLOR_RANGE[0],
        vmax=EFFICIENCY_COLOR_RANGE[1],
        origin="lower",
        aspect="auto",
    )
    axes.set_facecolor(SURFACE_COLOR)
    axes.set_xticks(range(len(BATCH_SIZES)), [str(batch_size) for batch_size in BATCH_SIZES])
    axes.set_yticks(range(len(CONTEXT_LENS)), [str(context_len) for context_len in CONTEXT_LENS])
    axes.tick_params(colors=MUTED_TEXT_COLOR, labelsize=8, length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def plot_efficiency_heatmaps(summaries: list[PointSummary]) -> Figure:
    figure = Figure(figsize=(12, 6), facecolor=SURFACE_COLOR)
    axes_grid = figure.subplots(
        len(KV_CACHE_DTYPES), len(COLOR_BY_MODEL), sharex=True, sharey=True, squeeze=False
    )
    for row_idx, kv_cache_dtype in enumerate(KV_CACHE_DTYPES):
        for column_idx, model in enumerate(COLOR_BY_MODEL):
            axes = axes_grid[row_idx, column_idx]
            panel_summaries = [
                summary
                for summary in summaries
                if summary.model == model and summary.kv_cache_dtype == kv_cache_dtype
            ]
            draw_efficiency_heatmap(axes, panel_summaries)
            axes.set_title(
                f"{to_model_label(model)}, {LABEL_BY_KV_CACHE_DTYPE[kv_cache_dtype]}",
                color=TEXT_COLOR,
                fontsize=9,
            )
    for axes in axes_grid[-1]:
        axes.set_xlabel("Batch size", color=TEXT_COLOR, fontsize=9)
    for axes in axes_grid[:, 0]:
        axes.set_ylabel("Context length (tokens)", color=TEXT_COLOR, fontsize=9)
    figure.suptitle(
        "Decode efficiency (predicted / measured step time)",
        color=TEXT_COLOR,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    figure.text(
        0.01,
        0.01,
        "* fewer than 3 repeats.  Blank: the KV cache did not fit under the 0.25 memory cap.",
        color=MUTED_TEXT_COLOR,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def plot_gpu_energy_per_token_vs_batch(summaries: list[PointSummary]) -> Figure:
    figure = Figure(figsize=(13, 4.2), facecolor=SURFACE_COLOR)
    axes_row = figure.subplots(1, len(CONTEXT_LENS), sharey=True)
    for axes, context_len in zip(axes_row, CONTEXT_LENS, strict=True):
        style_axes(axes)
        for model in COLOR_BY_MODEL:
            for kv_cache_dtype in KV_CACHE_DTYPES:
                draw_energy_line(
                    axes,
                    summaries,
                    model=model,
                    kv_cache_dtype=kv_cache_dtype,
                    context_len=context_len,
                )
        axes.set_xscale("log", base=2)
        axes.set_yscale("log")
        axes.set_xticks(BATCH_SIZES, [str(batch_size) for batch_size in BATCH_SIZES])
        axes.set_title(f"Context {context_len}", color=TEXT_COLOR, fontsize=9)
        axes.set_xlabel("Batch size", color=TEXT_COLOR, fontsize=9)
    axes_row[0].set_ylabel("GPU energy per token above idle (J)", color=TEXT_COLOR, fontsize=9)
    axes_row[0].legend(
        handles=build_model_and_dtype_legend_handles(
            model_linestyle="-", linestyle_by_kv_cache_dtype=LINESTYLE_BY_KV_CACHE_DTYPE
        ),
        fontsize=7,
        frameon=False,
        loc="lower left",
    )
    figure.suptitle(
        "GPU energy per decoded token falls with batch size",
        color=TEXT_COLOR,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


def draw_energy_line(
    axes: Axes, summaries: list[PointSummary], *, model: str, kv_cache_dtype: str, context_len: int
) -> None:
    batch_sizes: list[int] = []
    energies_j: list[float] = []
    for summary in summaries:
        is_on_line = (summary.model, summary.kv_cache_dtype, summary.context_len) == (
            model,
            kv_cache_dtype,
            context_len,
        )
        if not is_on_line:
            continue
        batch_sizes.append(summary.batch_size)
        energies_j.append(summary.gpu_energy_per_token_j_above_idle)
    color = COLOR_BY_MODEL[model]
    face_color = color if kv_cache_dtype == "auto" else SURFACE_COLOR
    axes.plot(
        batch_sizes,
        energies_j,
        linestyle=LINESTYLE_BY_KV_CACHE_DTYPE[kv_cache_dtype],
        marker="o",
        markersize=4,
        linewidth=2,
        color=color,
        markerfacecolor=face_color,
    )


def save_all_figures(summaries: list[PointSummary], plots_dir: Path) -> list[Path]:
    figure_by_name = {
        "measured_vs_predicted_tokens_per_s.png": plot_measured_vs_predicted_tokens_per_s(
            summaries
        ),
        "efficiency_heatmaps.png": plot_efficiency_heatmaps(summaries),
        "gpu_energy_per_token_vs_batch.png": plot_gpu_energy_per_token_vs_batch(summaries),
    }
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name, figure in figure_by_name.items():
        path = plots_dir / name
        figure.savefig(path, dpi=DPI, facecolor=SURFACE_COLOR)
        saved_paths.append(path)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw every figure from results/summary.csv.")
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for path in save_all_figures(load_point_summaries(args.summary), args.plots_dir):
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
