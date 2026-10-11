"""Monte Carlo convergence and variance reduction.

Backs the paper's Simulation noise paragraph; figure 5 is not in the paper.

For a handful of representative notes, prices each one at a ladder of path
counts under three generators - plain pseudo-random, antithetic variates and
scrambled Sobol - with 20 independent seeds (scrambles) per setting.  The
cross-seed standard deviation is the honest error of a single run for every
generator, including Sobol, whose per-run sample standard error is only an
iid-equivalent upper bound.

Experiment A: price, cross-seed std and 95% intervals against the path count,
with a reference value from 2^21 Sobol paths averaged over seeds.
Experiment B: variance-reduction ratio Var_plain / Var_method and wall time at
the production setting, 2^17 paths.

Notes come from the human-verified gold labels: a single-underlier note
(Citigroup, S&P 500), a two-asset worst-of (GS), a three-asset worst-of (TD),
plus the GS note with spot moved to 75% of its initial levels, just above the
70% coupon and knock-in barriers, where the discontinuous payoff converges
slowest.

Usage:
    python experiments/08_convergence.py               # full study
    python experiments/08_convergence.py --quick       # smoke run
    python experiments/08_convergence.py --plot-only   # redraw the convergence figure
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract.llm_schema import RawNoteTerms, to_note_terms
from src.market import build_market_inputs
from src.pricing import CommonRandomNumbers, price_note

GOLD = ROOT / "data" / "labels" / "gold"
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "paper" / "figures"

PATH_COUNTS = [2**13, 2**15, 2**17, 2**18, 2**19, 2**20]
PRODUCTION_N = 2**17
REFERENCE_N = 2**21
CHUNK = 2**16
METHODS = {
    "plain": dict(generator="iid", antithetic=False),
    "antithetic": dict(generator="iid", antithetic=True),
    "sobol": dict(generator="sobol", antithetic=False),
}

NOTES = [
    ("citi_single", "0000950103-26-010235", "Single (SPX)", None),
    ("gs_wo2", "0001193125-26-301771", "Worst-of 2 (RTY, SPX)", None),
    ("td_wo3", "0001140361-26-028280", "Worst-of 3 (DJI, NDX, SPX)", None),
    ("gs_wo2_near", "0001193125-26-301771", "Worst-of 2, spot at 75%", 0.75),
]


def load_note(accession: str, spot_fraction: float | None):
    payload = json.loads((GOLD / f"{accession}.json").read_text())
    terms, violations = to_note_terms(RawNoteTerms.model_validate(payload["terms"]))
    if terms is None:
        raise SystemExit(f"{accession}: gold label does not convert: {violations}")
    market = build_market_inputs(terms)
    if spot_fraction is not None:
        initial = np.array([u.initial_level for u in terms.underlyings], dtype=float)
        market = dataclasses.replace(market, spots=spot_fraction * initial)
    return terms, market


def run_one(terms, market, n_paths: int, seed: int, method: str):
    crn = CommonRandomNumbers(
        seed=seed,
        n_paths=n_paths,
        max_steps=terms.n_observations,
        max_assets=len(terms.underlyings),
        **METHODS[method],
    )
    started = time.perf_counter()
    result = price_note(terms, market, crn, chunk_paths=min(n_paths, CHUNK))
    return result, time.perf_counter() - started


def summarise(prices: list[float]) -> dict:
    arr = np.asarray(prices)
    std = float(arr.std(ddof=1))
    return {
        "mean": float(arr.mean()),
        "std": std,
        "ci95_single_run": 1.96 * std,
        "ci95_of_mean": 1.96 * std / np.sqrt(len(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw figure 5 from the saved convergence.json")
    args = parser.parse_args()

    if args.plot_only:
        out = json.loads((RESULTS / "convergence.json").read_text())
        labels = {key: label for key, _, label, _ in NOTES}
        for key, note in out["notes"].items():
            note["label"] = labels.get(key, note["label"])
        plot(out, FIGURES / "fig5_convergence.pdf")
        return 0

    path_counts = [2**11, 2**13] if args.quick else PATH_COUNTS
    reference_n = 2**15 if args.quick else REFERENCE_N
    production_n = 2**13 if args.quick else PRODUCTION_N
    seeds = list(range(1000, 1000 + args.seeds))

    out: dict = {
        "config": {
            "path_counts": path_counts,
            "reference_n": reference_n,
            "production_n": production_n,
            "seeds": seeds,
            "chunk_paths": CHUNK,
            "methods": METHODS,
        },
        "notes": {},
    }

    for key, accession, label, spot_fraction in NOTES:
        terms, market = load_note(accession, spot_fraction)
        print(f"== {label}  ({accession}, {terms.n_observations} obs)", flush=True)
        note: dict = {"label": label, "accession": accession, "spot_fraction": spot_fraction}

        ref = [run_one(terms, market, reference_n, s, "sobol")[0].price_per_1000 for s in seeds]
        note["reference"] = {"n_paths": reference_n, **summarise(ref)}
        print(f"   reference {note['reference']['mean']:.4f} "
              f"(std of mean {note['reference']['std'] / np.sqrt(len(seeds)):.4f})", flush=True)

        note["methods"] = {}
        for method in METHODS:
            rows = []
            for n in path_counts:
                prices, times, iid_se = [], [], []
                for s in seeds:
                    res, wall = run_one(terms, market, n, s, method)
                    prices.append(res.price_per_1000)
                    times.append(wall)
                    iid_se.append(res.std_error_per_1000)
                stats = summarise(prices)
                stats.update(
                    n_paths=n,
                    rmse_vs_reference=float(
                        np.sqrt(np.mean((np.asarray(prices) - note["reference"]["mean"]) ** 2))
                    ),
                    mean_reported_se=float(np.mean(iid_se)),
                    median_wall_s=float(np.median(times)),
                )
                rows.append(stats)
                print(f"   {method:10s} N={n:>9,}  mean {stats['mean']:9.4f}  "
                      f"std {stats['std']:.4f}  {stats['median_wall_s']:.3f}s", flush=True)
            note["methods"][method] = rows

        prod = {m: next(r for r in note["methods"][m] if r["n_paths"] == production_n)
                for m in METHODS}
        var_plain = prod["plain"]["std"] ** 2
        note["experiment_b"] = {
            "n_paths": production_n,
            "variance_ratio": {m: var_plain / prod[m]["std"] ** 2 for m in METHODS},
            "median_wall_s": {m: prod[m]["median_wall_s"] for m in METHODS},
            "sobol_cross_seed_se": prod["sobol"]["std"],
            "sobol_reported_iid_se": prod["sobol"]["mean_reported_se"],
        }
        print(f"   exp B: {note['experiment_b']['variance_ratio']}", flush=True)
        out["notes"][key] = note

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "convergence_quick.json" if args.quick else "convergence.json"
    (RESULTS / name).write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS / name}")
    plot(out, FIGURES / ("fig5_convergence_quick.pdf" if args.quick else "fig5_convergence.pdf"))
    return 0


def plot(out: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    # Categorical slots 1-3 of the reference palette (validated all-pairs, incl.
    # CVD); markers give a second, colour-free encoding for print.
    colors = {"plain": "#2a78d6", "antithetic": "#eb6834", "sobol": "#1baf7a"}
    names = {"plain": "Plain MC", "antithetic": "Antithetic", "sobol": "Scrambled Sobol"}
    markers = {"plain": "o", "antithetic": "s", "sobol": "^"}

    notes = list(out["notes"].values())
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.1), sharex=True)
    for ax, note in zip(axes.flat, notes):
        for method, rows in note["methods"].items():
            n = np.array([r["n_paths"] for r in rows], dtype=float)
            sd = np.array([r["std"] for r in rows])
            ax.plot(n, sd, marker=markers[method], ms=2.6, lw=1.0,
                    color=colors[method], label=names[method])
        plain = note["methods"]["plain"]
        n0, s0 = plain[0]["n_paths"], plain[0]["std"]
        n_ref = np.array([plain[0]["n_paths"], plain[-1]["n_paths"]], dtype=float)
        ax.plot(n_ref, s0 * np.sqrt(n0 / n_ref), ls=":", lw=0.8, color="#888888",
                label=r"$\propto 1/\sqrt{N}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(note["label"], pad=2, fontsize=6.5)
        ax.grid(True, which="major", lw=0.3, color="#dddddd")
    for ax in axes[1]:
        ax.set_xlabel("Paths $N$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cross-seed std ($/1000)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
