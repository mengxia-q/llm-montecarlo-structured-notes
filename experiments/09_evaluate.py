"""Score every arm against the reference labels (paper: Metrics).

Reference per filing: the human-verified gold label in ``data/labels/gold/`` if
there is one, else the Claude prelabel in ``data/labels/prelabel/`` (silver).
Every metric is reported for the full set and for the gold-only subset.

Metrics
  * field-level exact match on the critical fields (sequence fields count as
    correct only when the whole sequence matches), plus element-level F1 for
    the date schedule;
  * file-level all-critical-correct rate;
  * end-to-end price deviation Delta = V(extracted) - V(reference) per $1000,
    with common random numbers and the reference's market data;
  * coverage / selective accuracy as the confidence threshold tau sweeps 0..1.

Usage:
    python experiments/09_evaluate.py --run-id stage2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import price_pair
from src.extract.vote import field_value
from src.market import build_market_inputs
from src.schema import CRITICAL_FIELDS, NoteTerms

RESULTS = ROOT / "experiments" / "results"
LABELS = ROOT / "data" / "labels"
ARMS = ("B1", "B2", "B3", "Ours")

#: Field categories for the per-category breakdown (table 2, figure 3).
CATEGORY = {
    "observation_dates": "date schedule",
    "maturity_date": "date schedule",
    "autocall.levels_pct": "autocall terms",
    "autocall.first_observation_index": "autocall terms",
    "coupon.rate_per_period_pct": "coupon terms",
    "coupon.barrier_pct": "coupon terms",
    "coupon.memory": "coupon terms",
    "knock_in.barrier_pct": "knock-in barrier",
    "underlyings.initial_level": "initial levels",
}


def load_labels() -> dict[str, dict]:
    labels: dict[str, dict] = {}
    excluded = json.loads((LABELS / "exclusions.json").read_text())["excluded"]
    for source in ("prelabel", "gold"):  # gold overrides silver
        for path in sorted((LABELS / source).glob("*.json")):
            if path.stem in excluded:
                continue
            rec = json.loads(path.read_text())
            labels[path.stem] = {
                "terms": rec["terms"],
                "source": "gold" if source == "gold" else "silver",
                "scope": rec.get("scope"),
            }
    return labels


def norm(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return tuple(norm(v) for v in value)
    return value


#: Percent-valued fields match within this many percentage points.  Filings print
#: the per-period coupon both as a rate (e.g. 2.7725%) and as a cash amount
#: rounded to a tenth of a cent per $10 (e.g. $0.2773 = 2.773%); a reading of
#: either is a correct extraction.
PCT_TOLERANCE = 1e-3


def _close(a, b, tol: float) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= tol
    return norm(a) == norm(b)


def field_matches(pred: dict | None, ref: dict) -> dict[str, bool]:
    if pred is None:
        return {p: False for p in CRITICAL_FIELDS}
    out = {}
    for p in CRITICAL_FIELDS:
        a, b = field_value(pred, p), field_value(ref, p)
        out[p] = _close(a, b, PCT_TOLERANCE) if p.endswith("_pct") else norm(a) == norm(b)
    return out


def final_date_equivalent(pred: dict | None, ref: dict) -> bool:
    """True if the autocall ladders differ only in whether the final date is callable.

    Filings word this inconsistently ("any Observation Date" vs "interim
    valuation dates"), and the two readings price identically whenever the last
    call level is at or above the knock-in barrier: a final level above the call
    trigger is also above the barrier, so principal is repaid either way.
    """
    if pred is None:
        return False
    a = list(pred.get("autocall", {}).get("levels_pct") or [])
    b = list(ref.get("autocall", {}).get("levels_pct") or [])
    if abs(len(a) - len(b)) != 1:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    ki = ref.get("knock_in", {}).get("barrier_pct") or 0
    return _close(short, long_[:-1], PCT_TOLERANCE) and long_[-1] >= ki


def element_f1(pred: list | None, ref: list) -> float:
    pred_set, ref_set = set(pred or []), set(ref)
    if not pred_set and not ref_set:
        return 1.0
    tp = len(pred_set & ref_set)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred_set), tp / len(ref_set)
    return 2 * p * r / (p + r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-paths", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args()

    run_dir = RESULTS / "runs" / args.run_id
    arms = [json.loads(l) for l in (run_dir / "arms.jsonl").read_text().splitlines() if l]
    labels = load_labels()

    market_cache: dict[str, object] = {}
    rows: list[dict] = []
    for rec in arms:
        acc = rec["accession"]
        label = labels.get(acc)
        if label is None:
            continue
        if label.get("scope") and label["scope"].get("in_scope") is False:
            continue
        ref = label["terms"]
        matches = field_matches(rec["payload"], ref)
        pred_dates = (rec["payload"] or {}).get("observation_dates")
        row = {
            "accession": acc,
            "arm": rec["arm"],
            "label_source": label["source"],
            "schema_ok": rec["schema_ok"],
            "hard": len(rec["hard"]),
            "soft": len(rec["soft"]),
            "confidence": rec["confidence"],
            "matches": matches,
            "all_correct": all(matches.values()),
            "all_correct_norm": all(
                ok or (p == "autocall.levels_pct" and final_date_equivalent(rec["payload"], ref))
                for p, ok in matches.items()
            ),
            "dates_f1": element_f1(pred_dates, ref["observation_dates"]),
            "n_calls": rec["n_calls"],
            "cost_usd": rec["cost_usd"],
            "latency_s": rec["latency_s"],
            "delta": None,
            "priceable": False,
        }
        # Price deviation -------------------------------------------------
        try:
            gold_terms = NoteTerms.model_validate(ref)
        except Exception as exc:  # noqa: BLE001
            row["price_error"] = f"reference not priceable: {exc}"
            rows.append(row)
            continue
        if rec["schema_ok"] and rec["payload"] is not None:
            from src.extract import RawNoteTerms, to_note_terms

            ext_terms, _ = to_note_terms(RawNoteTerms.model_validate(rec["payload"]))
            try:
                if acc not in market_cache:
                    market_cache[acc] = build_market_inputs(gold_terms)
                market = market_cache[acc]
                ext_market = None
                if [u.name for u in ext_terms.underlyings] != [u.name for u in gold_terms.underlyings]:
                    ext_market = build_market_inputs(ext_terms)
                    row["basket_changed"] = True
                pair = price_pair(
                    gold_terms, ext_terms, market,
                    seed=args.seed, n_paths=args.n_paths, extracted_market=ext_market,
                )
                row["delta"] = pair.signed_deviation
                row["ref_price"] = pair.gold.price_per_1000
                row["priceable"] = True
            except Exception as exc:  # noqa: BLE001 - unpriceable extraction
                row["price_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    (run_dir / "eval_rows.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))

    summary: dict = {}
    for subset in ("all", "gold"):
        sub = [r for r in rows if subset == "all" or r["label_source"] == "gold"]
        summary[subset] = {}
        for arm in ARMS:
            ar = [r for r in sub if r["arm"] == arm]
            if not ar:
                continue
            n = len(ar)
            deltas = sorted(abs(r["delta"]) for r in ar if r["delta"] is not None)

            def q(p):
                if not deltas:
                    return None
                return deltas[min(len(deltas) - 1, int(math.ceil(p * len(deltas))) - 1)]

            per_field = {p: sum(r["matches"][p] for r in ar) / n for p in CRITICAL_FIELDS}
            per_cat: dict[str, list[bool]] = defaultdict(list)
            for r in ar:
                for p, ok in r["matches"].items():
                    per_cat[CATEGORY[p]].append(ok)
            summary[subset][arm] = {
                "n": n,
                "schema_fail_rate": sum(not r["schema_ok"] for r in ar) / n,
                "unpriceable_rate": sum(not r["priceable"] for r in ar) / n,
                "field_accuracy": sum(sum(r["matches"].values()) for r in ar) / (n * len(CRITICAL_FIELDS)),
                "per_field_accuracy": per_field,
                "per_category_accuracy": {k: sum(v) / len(v) for k, v in per_cat.items()},
                "file_all_correct": sum(r["all_correct"] for r in ar) / n,
                "file_all_correct_norm": sum(r["all_correct_norm"] for r in ar) / n,
                "dates_element_f1": sum(r["dates_f1"] for r in ar) / n,
                "delta_abs_mean": sum(deltas) / len(deltas) if deltas else None,
                "delta_abs_median": q(0.5),
                "delta_abs_p90": q(0.9),
                "delta_abs_max": deltas[-1] if deltas else None,
                "share_delta_gt_1": sum(d > 1 for d in deltas) / n,
                "share_delta_gt_10": sum(d > 10 for d in deltas) / n,
                "mean_calls": sum(r["n_calls"] for r in ar) / n,
                "mean_cost_usd": sum(r["cost_usd"] for r in ar) / n,
                "mean_latency_s": sum(r["latency_s"] for r in ar) / n,
            }
            if arm in ("B3", "Ours"):
                curve = []
                for tau in [i / 20 for i in range(21)]:
                    kept = [r for r in ar if (r["confidence"] or 0) >= tau]
                    curve.append(
                        {
                            "tau": tau,
                            "coverage": len(kept) / n,
                            "selective_accuracy": (
                                sum(r["all_correct"] for r in kept) / len(kept) if kept else None
                            ),
                            "selective_accuracy_norm": (
                                sum(r["all_correct_norm"] for r in kept) / len(kept) if kept else None
                            ),
                            "selective_delta_gt_1": (
                                sum(abs(r["delta"] or 0) > 1 or not r["priceable"] for r in kept) / len(kept)
                                if kept else None
                            ),
                        }
                    )
                summary[subset][arm]["coverage_curve"] = curve

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for subset, by_arm in summary.items():
        print(f"\n[{subset}]")
        print(f"  {'arm':5s} {'n':>3} {'field':>6} {'file':>6} {'fileN':>6} {'F1dt':>6} {'schema!':>7} "
              f"{'|D|med':>7} {'|D|p90':>7} {'|D|max':>8} {'>$1':>5} {'calls':>5} {'$/file':>7}")
        for arm, s in by_arm.items():
            fmt = lambda v, w=7, d=3: f"{v:{w}.{d}f}" if v is not None else " " * (w - 1) + "-"
            print(f"  {arm:5s} {s['n']:3d} {s['field_accuracy']:6.3f} {s['file_all_correct']:6.3f} {s['file_all_correct_norm']:6.3f} "
                  f"{s['dates_element_f1']:6.3f} {s['schema_fail_rate']:7.3f} {fmt(s['delta_abs_median'])} "
                  f"{fmt(s['delta_abs_p90'])} {fmt(s['delta_abs_max'], 8)} {s['share_delta_gt_1']:5.2f} "
                  f"{s['mean_calls']:5.2f} {s['mean_cost_usd']:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
