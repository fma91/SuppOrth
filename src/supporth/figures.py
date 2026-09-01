"""Protein-pair overlap figure written next to the consensus table.

The layout matches the proposed output minus the gene-level Venn: a four-set
(or fewer) Venn of protein pairs, a pairwise overlap heatmap, and a bar chart
of pairs unique to one predictor. Gene collapse is a separate table; it is not
plotted here.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .canonical import Pair, PredictorResult
from .consensus import tool_flag_columns

# Order and colours follow the proposed figure: OrthoFinder, Broccoli, OMA,
# SonicParanoid. Unknown tools are appended after these.
PANEL_ORDER = ("orthofinder", "broccoli", "oma", "sonicparanoid")
DISPLAY_NAME = {
    "orthofinder": "OrthoFinder",
    "broccoli": "Broccoli",
    "oma": "OMA",
    "sonicparanoid": "SonicParanoid",
}
SHORT_LABEL = {
    "orthofinder": "F",
    "broccoli": "B",
    "oma": "O",
    "sonicparanoid": "S",
}
SET_COLOR = {
    "orthofinder": "#6B5B95",
    "broccoli": "#5EB7B7",
    "oma": "#D46A7E",
    "sonicparanoid": "#5AA862",
}
FALLBACK_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


@dataclass(frozen=True)
class Exclusive:
    tool: str
    label: str
    count: int
    percent: float


@dataclass(frozen=True)
class OverlapReport:
    tools: tuple[str, ...]
    names: tuple[str, ...]
    labels: tuple[str, ...]
    colors: tuple[str, ...]
    sets: tuple[set[Pair], ...]
    matrix: tuple[tuple[int, ...], ...]
    exclusives: tuple[Exclusive, ...]

    def as_venn_sets(self) -> dict[str, set[int]]:
        """Integer ids only: venny4py writes every element to a scratch file."""
        assigned: dict[Pair, int] = {}
        compact: list[set[int]] = []
        for pairs in self.sets:
            ids: set[int] = set()
            for pair in pairs:
                ident = assigned.get(pair)
                if ident is None:
                    ident = len(assigned)
                    assigned[pair] = ident
                ids.add(ident)
            compact.append(ids)
        return dict(zip(self.names, compact))


def overlap_report(results: Sequence[PredictorResult]) -> OverlapReport:
    """Pairwise intersections and exclusive counts from unfiltered pair sets."""
    by_tool: dict[str, PredictorResult] = {}
    extras: list[str] = []
    for result in results:
        if not result.pairs:
            continue
        if result.tool in by_tool:
            by_tool[result.tool].pairs.update(result.pairs)
            continue
        by_tool[result.tool] = PredictorResult(
            result.tool, result.label, set(result.pairs)
        )
        if result.tool not in PANEL_ORDER:
            extras.append(result.tool)

    tools = [name for name in PANEL_ORDER if name in by_tool] + extras
    names: list[str] = []
    labels: list[str] = []
    colors: list[str] = []
    sets: list[set[Pair]] = []
    for index, tool in enumerate(tools):
        result = by_tool[tool]
        names.append(DISPLAY_NAME.get(tool, result.tool))
        labels.append(SHORT_LABEL.get(tool, result.label or result.tool[:1].upper()))
        colors.append(SET_COLOR.get(tool, FALLBACK_COLORS[index % len(FALLBACK_COLORS)]))
        sets.append(set(result.pairs))

    matrix = tuple(
        tuple(len(left & right) for right in sets) for left in sets
    )
    exclusives = []
    for index, tool in enumerate(tools):
        others: set[Pair] = set()
        for j, pairs in enumerate(sets):
            if j != index:
                others |= pairs
        only = sets[index] - others
        total = len(sets[index])
        exclusives.append(
            Exclusive(
                tool=tool,
                label=labels[index],
                count=len(only),
                percent=(100.0 * len(only) / total) if total else 0.0,
            )
        )
    return OverlapReport(
        tools=tuple(tools),
        names=tuple(names),
        labels=tuple(labels),
        colors=tuple(colors),
        sets=tuple(sets),
        matrix=matrix,
        exclusives=tuple(exclusives),
    )


def results_from_table(frame) -> list[PredictorResult]:
    """Rebuild pair sets from a consensus TSV (unfiltered or filtered)."""
    sp1, sp2 = frame.columns[0], frame.columns[1]
    results = []
    for tool in tool_flag_columns(frame):
        mask = frame[tool].fillna(0).astype(int) == 1
        pairs = set(zip(frame.loc[mask, sp1], frame.loc[mask, sp2]))
        results.append(
            PredictorResult(
                tool=tool,
                label=SHORT_LABEL.get(tool, str(tool)[:1].upper()),
                pairs=pairs,
            )
        )
    return results


def write_support_figure(
    results: Sequence[PredictorResult],
    destination: Path,
    *,
    log: Callable[[str], None] = print,
) -> Path | None:
    """Write the protein-pair figure. Returns None if plotting is unavailable."""
    report = overlap_report(results)
    if len(report.tools) < 2:
        log("  skip figure: need at least two predictors with pairs")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        log("  skip figure: matplotlib is not installed (pip install matplotlib venny4py)")
        return None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.2, 7.0), facecolor="white")
    fig.suptitle(
        "Predicted orthologue pairs toward support creation",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.2, 1], wspace=0.32, hspace=0.38,
        left=0.04, right=0.96, top=0.90, bottom=0.08,
    )
    ax_venn = fig.add_subplot(grid[:, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[1, 1])

    _draw_venn(ax_venn, report)
    _draw_heatmap(ax_heat, report, Normalize)
    _draw_exclusive_bars(ax_bar, report)

    png = destination.with_suffix(".png")
    svg = destination.with_suffix(".svg")
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    log(f"  figure: {png}")
    log(f"  figure: {svg}")
    return png


def _draw_venn(ax, report: OverlapReport) -> None:
    ax.set_title("By protein pairs", loc="left", fontsize=11, pad=8)
    ax.set_axis_off()
    if not (2 <= len(report.sets) <= 4):
        ax.text(0.5, 0.5, "Venn is drawn for 2–4 predictors", ha="center", va="center")
        return
    try:
        from venny4py.venny4py import venny4py
    except ImportError:
        ax.text(
            0.5, 0.5,
            "venny4py is not installed\n(pip install venny4py)",
            ha="center", va="center",
        )
        return
    with tempfile.TemporaryDirectory() as scratch:
        venny4py(
            sets=report.as_venn_sets(),
            out=scratch,
            asax=ax,
            size=3.8,
            colors=report.colors,
            legend_cols=2,
            font_size=9,
            line_width=0.6,
            alpha=0.28,
        )


def _draw_heatmap(ax, report: OverlapReport, Normalize) -> None:
    import numpy as np

    values = np.array(report.matrix, dtype=float)
    rows, cols = values.shape
    mesh = ax.pcolormesh(
        np.arange(cols + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        values,
        cmap="Blues",
        norm=Normalize(vmin=0, vmax=values.max() or 1),
        edgecolors="white",
        linewidth=0.4,
        antialiased=True,
    )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Pairwise overlap between tools", fontsize=11, pad=8)
    ticks = range(len(report.names))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(report.names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(report.names, fontsize=8)
    cutoff = values.max() * 0.6
    for row, row_values in enumerate(report.matrix):
        for col, value in enumerate(row_values):
            ax.text(
                col, row, f"{value:,}",
                ha="center", va="center", fontsize=8,
                color="white" if value > cutoff else "black",
            )
    ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)


def _draw_exclusive_bars(ax, report: OverlapReport) -> None:
    labels = [item.label for item in report.exclusives]
    counts = [item.count for item in report.exclusives]
    percents = [item.percent for item in report.exclusives]
    colors = list(report.colors)
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.72)
    ax.set_title("Unique orthologue pairs", fontsize=11, pad=8)
    ax.set_ylabel("Unique orthologue pairs")
    ax.set_xlabel("")
    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, percent in zip(bars, percents):
        label = f"{percent:.1f}%" if percent < 10 else f"{percent:.0f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center", va="bottom", fontsize=8,
        )
