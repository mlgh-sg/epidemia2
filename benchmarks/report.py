#!/usr/bin/env python
"""Turn benchmarks/run.py output into the figures and tables used in the docs.

    uv run --project python python benchmarks/report.py \
        --json $TMPDIR/epidemia-bench/benchmark.json \
        --out python/docs/img

Produces three figures and a markdown table:

  bench-speed.png       sampling wall-clock per engine
  bench-efficiency.png  effective samples per second -- the metric that matters,
                        since a fast sampler that mixes badly is not fast
  bench-agreement.png   the five NPI effects estimated by each engine, which is
                        how you tell the two implementations are the same model
  benchmark-table.md    every number, for readers who want them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Validated categorical slots (see the data-viz palette reference). Assigned in
# fixed order to engines, so an engine keeps its colour across every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8981"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"

NPI_LABEL = {
    "public_events": "Public events",
    "schools_universities": "Schools",
    "self_isolating_if_ill": "Self-isolation",
    "social_distancing_encouraged": "Distancing",
    "lockdown": "Lockdown",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _rounded_bars(ax, ys, widths, colors, height=0.46):
    """Horizontal bars with a 4px rounded data-end, anchored at the baseline."""
    for y, w, c in zip(ys, widths, colors):
        # radius in data units, capped so short bars do not become lozenges
        r = min(w * 0.06, height / 2)
        ax.add_patch(
            FancyBboxPatch(
                (0, y - height / 2), max(w - r, 1e-9), height,
                boxstyle=f"round,pad=0,rounding_size={r}",
                linewidth=0, facecolor=c, mutation_aspect=1,
            )
        )


def bar_figure(labels, values, colors, title, subtitle, xlabel, fmt, path):
    fig, ax = plt.subplots(figsize=(7.6, 2.6 + 0.3 * len(labels)))
    ys = list(range(len(labels)))[::-1]
    _rounded_bars(ax, ys, values, colors)

    span = max(values) if values else 1
    for y, v in zip(ys, values):
        ax.text(v + span * 0.02, y, fmt(v), va="center", ha="left",
                fontsize=10, color=INK, fontweight="medium")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    # headroom for the direct labels, which sit outside the bar end
    ax.set_xlim(0, span * 1.34)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2, labelpad=8)
    _style(ax)

    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=22,
                 fontweight="semibold")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK_2, va="bottom")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def agreement_figure(results, path):
    """Effect estimates per engine: same model => the intervals should overlap."""
    npis = list(NPI_LABEL)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    n = len(results)
    offsets = [(i - (n - 1) / 2) * 0.22 for i in range(n)]

    for i, (res, off) in enumerate(zip(results, offsets)):
        colour = SERIES[i % len(SERIES)]
        eff = res["effects"]
        keys = list(eff)
        for j, npi in enumerate(npis):
            key = npi if npi in eff else (keys[j] if j < len(keys) else None)
            if key is None:
                continue
            m, s = eff[key]["mean"], eff[key]["sd"]
            y = len(npis) - 1 - j + off
            ax.plot([m - 1.96 * s, m + 1.96 * s], [y, y],
                    color=colour, linewidth=2, solid_capstyle="round",
                    zorder=2)
            ax.plot([m], [y], marker="o", markersize=8, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                    label=res["engine"] if j == 0 else None)

    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(npis)))
    ax.set_yticklabels([NPI_LABEL[n] for n in npis][::-1], fontsize=10, color=INK)
    ax.set_xlabel("Effect on the logit of $R_t$   (negative = reduces transmission)",
                  fontsize=9, color=INK_2, labelpad=8)
    _style(ax)
    ax.yaxis.grid(False)

    # Title sits above the subtitle block, which grows upward from y=1.02.
    ax.set_title("Consistent, within Monte Carlo error", fontsize=13, color=INK,
                 loc="left", pad=78, fontweight="semibold")
    ax.text(0, 1.02,
            "Posterior mean and 95% interval per NPI. Every interval overlaps\n"
            "across engines; the collinear effects trade off, so the point\n"
            "estimates differ by up to about one posterior SD.",
            transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom",
            linespacing=1.45)

    # Below the axes: inside the panel this collides with the bottom row.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    frameon=False, fontsize=9, ncol=3, handletextpad=0.5,
                    columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def markdown_table(payload, path):
    rows = payload["results"]
    cfg = payload["config"]
    m = payload["machine"]
    v = payload["versions"]

    lines = [
        "| Engine | Compile (s) | Sample (s) | min ESS | ESS/s | Divergences | max R-hat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ess_s = r["min_ess_bulk"] / r["sample_seconds"] if r["sample_seconds"] else float("nan")
        lines.append(
            f"| {r['engine']} | {r['compile_seconds']:.1f} | {r['sample_seconds']:.1f} | "
            f"{r['min_ess_bulk']:.0f} | {ess_s:.2f} | "
            f"{'--' if r['divergences'] is None else r['divergences']} | "
            f"{r['max_rhat']:.3f} |"
        )
    lines += [
        "",
        f"{cfg['chains']} chains x ({cfg['tune']} tune + {cfg['draws']} draws), "
        f"seed {cfg['seed']}. Model: {cfg['model']}.",
        "",
        f"Run on {m['processor']}, {m['cpu_count']} cores, {m['platform']}. "
        f"Python {m['python']}; "
        + ", ".join(f"{k} {val}" for k, val in v.items() if val),
        "",
        "min ESS is the smallest bulk effective sample size across the five NPI "
        "effects; ESS/s divides it by sampling wall-clock. Compile time is "
        "excluded from ESS/s because Stan caches its executable across runs "
        "while nutpie re-compiles each time.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.json.read_text())
    results = payload["results"]
    labels = [r["engine"] for r in results]
    colors = [SERIES[i % len(SERIES)] for i in range(len(results))]

    bar_figure(
        labels, [r["sample_seconds"] for r in results], colors,
        "Sampling wall-clock",
        f"{payload['config']['chains']} chains x "
        f"({payload['config']['tune']} tune + {payload['config']['draws']} draws), "
        "run one after another on an idle machine. Lower is better.",
        "seconds", lambda v: f"{v:,.0f}s", args.out / "bench-speed.png",
    )

    bar_figure(
        labels,
        [r["min_ess_bulk"] / r["sample_seconds"] for r in results], colors,
        "Effective samples per second",
        "What actually matters: a fast sampler that mixes badly is not fast. "
        "Higher is better.",
        "min bulk ESS / second", lambda v: f"{v:.2f}",
        args.out / "bench-efficiency.png",
    )

    agreement_figure(results, args.out / "bench-agreement.png")
    markdown_table(payload, args.out.parent / "benchmark-table.md")


if __name__ == "__main__":
    main()
