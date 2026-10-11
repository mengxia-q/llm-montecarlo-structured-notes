"""Per-field price sensitivity by controlled perturbation, over every labelled filing.

Feeds the paper's Which Errors Matter section and figure 3.

Each critical field of the reference parameter set is perturbed one at a time
with a realistic extraction error (the perturbations of ``02_vertical_slice.py``)
and priced against the unperturbed set with common random numbers.  Also checks
seed stability: for every arm/filing with a non-zero Delta in
``eval_rows.jsonl``, Delta is recomputed under 20 seeds to show that it does not
depend on the seed.

Usage:
    python experiments/10_sensitivity.py --run-id stage2
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import price_pair
from src.extract import RawNoteTerms, to_note_terms
from src.market import build_market_inputs
from src.schema import NoteTerms

RESULTS = ROOT / "experiments" / "results"

_spec = importlib.util.spec_from_file_location("slice", ROOT / "experiments" / "02_vertical_slice.py")
_slice = importlib.util.module_from_spec(_spec)
sys.modules["slice"] = _slice
_spec.loader.exec_module(_slice)

_evaluate = importlib.util.spec_from_file_location("ev", ROOT / "experiments" / "09_evaluate.py")
_ev = importlib.util.module_from_spec(_evaluate)
sys.modules["ev"] = _ev
_evaluate.loader.exec_module(_ev)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-paths", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    run_dir = RESULTS / "runs" / args.run_id

    labels = _ev.load_labels()
    sens: dict[str, list[float]] = {}
    per_filing: list[dict] = []
    markets: dict[str, object] = {}
    for acc, label in sorted(labels.items()):
        if label.get("scope") and label["scope"].get("in_scope") is False:
            continue
        try:
            ref = NoteTerms.model_validate(label["terms"])
            markets[acc] = market = build_market_inputs(ref)
        except Exception as exc:  # noqa: BLE001
            print(f"  {acc}: skipped ({type(exc).__name__}: {exc})")
            continue
        row = {"accession": acc, "label_source": label["source"], "deltas": {}}
        for pert in _slice._perturbations():
            dump = copy.deepcopy(ref.model_dump(mode="json"))
            pert.apply(dump)
            try:
                perturbed = NoteTerms.model_validate(dump)
            except Exception:  # noqa: BLE001 - perturbation invalid for this filing
                continue
            if perturbed.model_dump() == ref.model_dump():
                continue
            pair = price_pair(ref, perturbed, market, seed=args.seed, n_paths=args.n_paths)
            row["deltas"][pert.name] = pair.signed_deviation
            row["ref_price"] = pair.gold.price_per_1000
            sens.setdefault(pert.name, []).append(abs(pair.signed_deviation))
        per_filing.append(row)
        print(f"  {acc} {len(row['deltas'])} perturbations", flush=True)

    ranking = sorted(
        (
            {
                "field": k,
                "description": next(p.description for p in _slice._perturbations() if p.name == k),
                "median_abs": statistics.median(v),
                "mean_abs": statistics.fmean(v),
                "max_abs": max(v),
                "n": len(v),
            }
            for k, v in sens.items()
        ),
        key=lambda r: r["median_abs"],
        reverse=True,
    )

    # Experiment C: Delta stability across seeds ---------------------------
    stability: list[dict] = []
    rows_path = run_dir / "eval_rows.jsonl"
    arms = {
        (r["accession"], r["arm"]): r
        for r in map(json.loads, (run_dir / "arms.jsonl").read_text().splitlines())
    }
    if rows_path.exists():
        for row in map(json.loads, rows_path.read_text().splitlines()):
            if not row.get("delta") or abs(row["delta"]) < 1e-9 or row["accession"] not in markets:
                continue
            ref = NoteTerms.model_validate(labels[row["accession"]]["terms"])
            ext, _ = to_note_terms(RawNoteTerms.model_validate(arms[(row["accession"], row["arm"])]["payload"]))
            if row.get("basket_changed"):
                continue
            ds = [
                price_pair(ref, ext, markets[row["accession"]], seed=args.seed + s, n_paths=args.n_paths).signed_deviation
                for s in range(args.n_seeds)
            ]
            stability.append(
                {
                    "accession": row["accession"],
                    "arm": row["arm"],
                    "delta_seed0": row["delta"],
                    "mean": statistics.fmean(ds),
                    "std": statistics.stdev(ds),
                    "min": min(ds),
                    "max": max(ds),
                }
            )

    out = {"ranking": ranking, "per_filing": per_filing, "stability": stability}
    (run_dir / "sensitivity.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'field':36s} {'median':>9} {'mean':>9} {'max':>9} {'n':>3}")
    for r in ranking:
        print(f"{r['field']:36s} {r['median_abs']:9.3f} {r['mean_abs']:9.3f} {r['max_abs']:9.3f} {r['n']:3d}")
    if stability:
        print("\nDelta stability across seeds (exp. C):")
        for s in stability:
            print(f"  {s['accession']} {s['arm']:4s} delta {s['delta_seed0']:+9.3f}  mean {s['mean']:+9.3f}  std {s['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
