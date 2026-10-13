"""Robustness of the findings to the pricing template's market assumptions (revision, reviewer point 3).

The template uses 252-day historical volatility and correlation, a proxy-ETF
dividend yield and the Treasury zero curve.  Each scenario below replaces one of
these choices; both valuations of every pair use the replaced market state, as
in the base case, so Delta still isolates extraction error.  Per scenario we
recompute

  * the per-field sensitivity ranking of ``10_sensitivity.py`` (figure 3), and
    its Spearman rank correlation with the base ranking;
  * Delta for every arm/filing of the given runs whose base Delta is non-zero
    (a zero Delta comes from identical priced parameters and stays zero), and
    the resulting max |Delta| per arm.

Usage:
    python experiments/19_model_sensitivity.py --run-ids stage2 stage2_nothink
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import price_pair
from src.extract import RawNoteTerms, to_note_terms
from src.market import build_market_inputs
from src.pricing.market import ZeroCurve
from src.schema import NoteTerms

RESULTS = ROOT / "experiments" / "results"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_slice = _load("slice", ROOT / "experiments" / "02_vertical_slice.py")
_ev = _load("ev", ROOT / "experiments" / "09_evaluate.py")


def _shift_curve(curve: ZeroCurve, bp: float) -> ZeroCurve:
    return ZeroCurve(tenors=curve.tenors, rates=tuple(r + bp / 1e4 for r in curve.rates))


def _corr_toward(corr: np.ndarray, target: float, w: float) -> np.ndarray:
    """Convex combination with the constant-correlation matrix, which stays PSD."""
    n = corr.shape[0]
    tgt = np.full((n, n), target)
    np.fill_diagonal(tgt, 1.0)
    return (1 - w) * corr + w * tgt


#: name -> (description, build-time window or None, transform of MarketInputs)
SCENARIOS = {
    "base": ("252-day hist. vol/corr, proxy dividends, Treasury curve", None, lambda m: m),
    "window_126": ("126-day estimation window", 126, lambda m: m),
    "window_378": ("378-day estimation window", 378, lambda m: m),
    "vol_x0.8": ("volatilities x 0.8", None, lambda m: dataclasses.replace(m, vols=m.vols * 0.8)),
    "vol_x1.25": ("volatilities x 1.25", None, lambda m: dataclasses.replace(m, vols=m.vols * 1.25)),
    "corr_low": ("correlations halved", None, lambda m: dataclasses.replace(m, corr=_corr_toward(m.corr, 0.0, 0.5))),
    "corr_high": ("correlations halfway to 1", None, lambda m: dataclasses.replace(m, corr=_corr_toward(m.corr, 1.0, 0.5))),
    "div_zero": ("no dividends", None, lambda m: dataclasses.replace(m, div_yields=m.div_yields * 0.0)),
    "div_x1.5": ("dividend yields x 1.5", None, lambda m: dataclasses.replace(m, div_yields=m.div_yields * 1.5)),
    "rate_-100bp": ("zero curve -100 bp", None, lambda m: dataclasses.replace(m, curve=_shift_curve(m.curve, -100))),
    "rate_+100bp": ("zero curve +100 bp", None, lambda m: dataclasses.replace(m, curve=_shift_curve(m.curve, 100))),
}


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs="+", default=["stage2", "stage2_nothink"])
    parser.add_argument("--n-paths", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--out", default=str(RESULTS / "model_sensitivity.json"))
    args = parser.parse_args()

    labels = _ev.load_labels()
    refs: dict[str, NoteTerms] = {}
    for acc, label in sorted(labels.items()):
        if label.get("scope") and label["scope"].get("in_scope") is False:
            continue
        refs[acc] = NoteTerms.model_validate(label["terms"])

    perturbed: dict[str, dict[str, NoteTerms]] = {}
    for acc, ref in refs.items():
        perturbed[acc] = {}
        for pert in _slice._perturbations():
            dump = copy.deepcopy(ref.model_dump(mode="json"))
            pert.apply(dump)
            try:
                p = NoteTerms.model_validate(dump)
            except Exception:  # noqa: BLE001 - perturbation invalid for this filing
                continue
            if p.model_dump() != ref.model_dump():
                perturbed[acc][pert.name] = p

    # Extractions with a non-zero base Delta, per run and arm.
    extractions: list[dict] = []
    for run_id in args.run_ids:
        run_dir = RESULTS / "runs" / run_id
        arms = {(r["accession"], r["arm"]): r for r in map(json.loads, (run_dir / "arms.jsonl").read_text().splitlines())}
        for row in map(json.loads, (run_dir / "eval_rows.jsonl").read_text().splitlines()):
            if row.get("delta") is None or abs(row["delta"]) < 1e-9 or row["accession"] not in refs:
                continue
            if row.get("basket_changed"):
                continue
            ext, _ = to_note_terms(RawNoteTerms.model_validate(arms[(row["accession"], row["arm"])]["payload"]))
            extractions.append({"run_id": run_id, "accession": row["accession"], "arm": row["arm"],
                                "base_delta": row["delta"], "terms": ext})

    out: dict[str, dict] = {}
    for name, (desc, window, transform) in SCENARIOS.items():
        markets = {}
        for acc, ref in refs.items():
            m = build_market_inputs(ref, window=window) if window else build_market_inputs(ref)
            markets[acc] = transform(m)
        sens: dict[str, list[float]] = {}
        for acc, perts in perturbed.items():
            for pname, p in perts.items():
                pair = price_pair(refs[acc], p, markets[acc], seed=args.seed, n_paths=args.n_paths)
                sens.setdefault(pname, []).append(abs(pair.signed_deviation))
        ranking = {k: {"median_abs": statistics.median(v), "max_abs": max(v), "n": len(v)} for k, v in sens.items()}
        deltas = []
        for e in extractions:
            d = price_pair(refs[e["accession"]], e["terms"], markets[e["accession"]],
                           seed=args.seed, n_paths=args.n_paths).signed_deviation
            deltas.append({k: e[k] for k in ("run_id", "accession", "arm", "base_delta")} | {"delta": d})
        out[name] = {"description": desc, "ranking": ranking, "deltas": deltas}
        top = sorted(ranking, key=lambda k: ranking[k]["median_abs"], reverse=True)
        print(f"{name:12s} top {top[0]} ({ranking[top[0]]['median_abs']:.2f}), "
              f"2nd {top[1]} ({ranking[top[1]]['median_abs']:.2f})", flush=True)

    fields = sorted(out["base"]["ranking"])
    base_med = [out["base"]["ranking"][f]["median_abs"] for f in fields]
    summary = []
    for name, res in out.items():
        med = [res["ranking"][f]["median_abs"] for f in fields]
        order = sorted(fields, key=lambda f: res["ranking"][f]["median_abs"], reverse=True)
        maxd: dict[str, float] = {}
        for d in res["deltas"]:
            key = f"{d['run_id']}/{d['arm']}"
            maxd[key] = max(maxd.get(key, 0.0), abs(d["delta"]))
        summary.append({
            "scenario": name,
            "description": res["description"],
            "spearman_vs_base": spearman(base_med, med),
            "top_field": order[0],
            "top_median": res["ranking"][order[0]]["median_abs"],
            "second_field": order[1],
            "top_over_second": res["ranking"][order[0]]["median_abs"] / res["ranking"][order[1]]["median_abs"],
            "order": order,
            "max_abs_delta": maxd,
        })
    Path(args.out).write_text(json.dumps({"summary": summary, "scenarios": out}, indent=2))

    print(f"\n{'scenario':12s} {'rho':>5} {'top':>28} {'top/2nd':>8}  max|Delta| per run/arm")
    for s in summary:
        md = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(s["max_abs_delta"].items()))
        print(f"{s['scenario']:12s} {s['spearman_vs_base']:5.2f} {s['top_field']:>28} {s['top_over_second']:8.1f}  {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
