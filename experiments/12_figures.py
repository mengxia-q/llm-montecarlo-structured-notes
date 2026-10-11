"""Figures 2-4 for the paper, as vector PDFs sized for an IEEE single column.

    fig2_deviation.pdf   |Delta| per filing and arm, reasoning on vs off
    fig3_sensitivity.pdf median |Delta| per perturbed field
    fig4_coverage.pdf    coverage vs selective accuracy

Usage:
    python experiments/12_figures.py --think stage2 --nothink stage2_nothink
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "results" / "runs"
OUT = ROOT / "paper" / "figures"

#: Categorical slots 1-4 of the validated reference palette (light mode).
ARM_COLORS = {"B1": "#2a78d6", "B2": "#eb6834", "B3": "#1baf7a", "Ours": "#eda100"}
ARM_MARKERS = {"B1": "o", "B2": "s", "B3": "^", "Ours": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
FIELD_LABELS = {
    "coupon.rate_per_period_pct": "coupon rate (p.a. as per-period)",
    "coupon.barrier_pct": "coupon barrier +5pp",
    "autocall.levels_pct": "autocall level +5pp",
    "knock_in.monitoring": "knock-in monitoring flipped",
    "knock_in.barrier_pct": "knock-in barrier −5pp",
    "autocall.first_observation_index": "first call date one early",
    "coupon.memory": "memory flag flipped",
    "underlyings.initial_level": "initial level +1%",
    "observation_dates": "final observation dropped",
}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=3, labelsize=7)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def rows(run: str) -> list[dict]:
    return [json.loads(l) for l in (RUNS / run / "eval_rows.jsonl").read_text().splitlines()]


def fig2(think: str, nothink: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=True)
    floor = 1e-3
    for ax, run, title in zip(axes, (think, nothink), ("reasoning on", "reasoning off")):
        rs = rows(run)
        for i, arm in enumerate(ARM_COLORS):
            vals = [abs(r["delta"]) for r in rs if r["arm"] == arm and r["delta"] is not None]
            n_fail = sum(1 for r in rs if r["arm"] == arm and r["delta"] is None)
            v = np.maximum(np.array(vals), floor)
            jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35
            ax.scatter(i + jitter, v, s=9, marker=ARM_MARKERS[arm], color=ARM_COLORS[arm],
                       edgecolor="white", linewidth=0.4, zorder=3)
            if n_fail:
                ax.scatter([i] * n_fail, [1.2e3] * n_fail, s=14, marker="x", color=ARM_COLORS[arm],
                           linewidth=1.0, zorder=3)
        ax.axhline(4e2, color=GRID, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 3e3)
        ax.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xticks(range(4), list(ARM_COLORS), fontsize=7)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(r"$|\Delta|$ per \$1,000", fontsize=7, color=INK)
    axes[1].text(3.45, 1.15, "$1", fontsize=6, color=MUTED, ha="right", va="bottom")
    axes[0].set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1.2e3],
                       [r"$\leq\!10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "10", "100", "unpriced"])
    axes[0].tick_params(axis="y", which="minor", left=False)
    axes[1].tick_params(axis="y", which="minor", left=False)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "fig2_deviation.pdf")
    plt.close(fig)


def fig3(run: str) -> None:
    ranking = json.loads((RUNS / run / "sensitivity.json").read_text())["ranking"]
    ranking = sorted(ranking, key=lambda r: r["median_abs"])
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(len(ranking))
    med = [r["median_abs"] for r in ranking]
    lo = [min(r["median_abs"], r["median_abs"]) for r in ranking]
    mx = [r["max_abs"] for r in ranking]
    ax.hlines(y, [m for m in med], mx, color=GRID, linewidth=2, zorder=1)
    ax.scatter(med, y, s=18, color=ARM_COLORS["B1"], zorder=3, label="median")
    ax.scatter(mx, y, s=10, facecolor="white", edgecolor=ARM_COLORS["B1"], linewidth=0.8, zorder=3, label="max")
    for yi, m in zip(y, med):
        ax.text(m / 1.35, yi, f"{m:.2f}" if m < 100 else f"{m:.0f}", ha="right", va="center", fontsize=6, color=INK)
    ax.set_yticks(y, [FIELD_LABELS.get(r["field"], r["field"]) for r in ranking], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 2e3)
    ax.set_xlabel(r"$|\Delta|$ per \$1,000, across 50 filings", fontsize=7, color=INK)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig3_sensitivity.pdf")
    plt.close(fig)


def fig4(think: str, nothink: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    series = [
        (think, "selective_accuracy", "on, exact", "-", ARM_COLORS["B1"], "o"),
        (nothink, "selective_accuracy", "off, exact", "-", ARM_COLORS["B2"], "s"),
        (think, "selective_accuracy_norm", "on, price-equiv.", "--", ARM_COLORS["B1"], "o"),
        (nothink, "selective_accuracy_norm", "on/off, price-equiv.", "--", ARM_COLORS["B2"], "s"),
    ]
    for run, key, label, ls, color, mk in series:
        curve = json.loads((RUNS / run / "summary.json").read_text())["all"]["Ours"]["coverage_curve"]
        pts = sorted({(c["coverage"], c[key]) for c in curve if c[key] is not None})
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=color, linewidth=1.4, marker=mk, markersize=4,
                markeredgecolor="white", markeredgewidth=0.5)
        # The on/price-equiv. curve is 1 at every threshold, hidden under the off curve.
        pos = {"on, exact": (0.985, 0.905, "right"), "off, exact": (0.48, ys[0], "right"),
               "on/off, price-equiv.": (0.52, 0.985, "left")}.get(label)
        if pos is None:
            continue
        lx, ly, ha = pos
        ax.text(lx, ly, label, fontsize=6, color=INK, ha=ha, va="center" if label == "off, exact" else "top")
    ax.set_xlim(0.3, 1.03)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("coverage (share processed automatically)", fontsize=7, color=INK)
    ax.set_ylabel("file-level accuracy", fontsize=7, color=INK)
    style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig4_coverage.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--think", default="stage2")
    parser.add_argument("--nothink", default="stage2_nothink")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    fig2(args.think, args.nothink)
    fig3(args.nothink)
    fig4(args.think, args.nothink)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("fig*.pdf"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
