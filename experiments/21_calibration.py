"""Per-sample scoring: seed replication and confidence calibration (revision, reviewer points 4 and 5).

Every individual sample of every given run is scored as if it were B1: field
matches against the reference, the hard rules, and Delta under common random
numbers.  Each sample also gets a *pre-vote* agreement: for each critical field,
the share of the filing's k samples that propose the same value as this sample,
and c_sample = the minimum over critical fields.  This asks the question the
paper's coverage analysis could not: when a single draw is wrong in a way that
costs money, would agreement with the other draws (or a hard rule) have flagged
it before the vote removed it?

An error is *material* if the sample cannot be priced or |Delta| >= $1 per $1,000.

Outputs ``experiments/results/calibration.json``: per-sample records, a per-run
B1 replicate table, and detection rates / AUROC of 1 - c_sample for material
errors, overall and per run group.

Usage:
    python experiments/21_calibration.py --run-ids stage2 stage2_rep1 ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import price_pair
from src.extract import RawNoteTerms, to_note_terms
from src.extract.vote import _hashable, field_value
from src.market import build_market_inputs
from src.schema import CRITICAL_FIELDS, NoteTerms

RESULTS = ROOT / "experiments" / "results"
MATERIAL_USD = 1.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ev = _load("ev", ROOT / "experiments" / "09_evaluate.py")
_arms = _load("arms", ROOT / "experiments" / "07_run_arms.py")


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """P(score of a material error > score of a non-material sample), ties count 1/2."""
    if not scores_pos or not scores_neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in scores_pos for n in scores_neg)
    return wins / (len(scores_pos) * len(scores_neg))


def score_run(run_id: str, labels: dict, markets: dict, args) -> list[dict]:
    run_dir = RESULTS / "runs" / run_id
    calls = [json.loads(l) for l in (run_dir / "calls.jsonl").read_text().splitlines() if l]
    by_acc: dict[str, dict[int, dict]] = defaultdict(dict)
    for c in calls:
        if c["kind"].startswith("sample_"):
            by_acc[c["accession"]][int(c["kind"].split("_")[1])] = c  # last record wins
    out = []
    for acc, samples in sorted(by_acc.items()):
        label = labels.get(acc)
        if label is None or (label.get("scope") and label["scope"].get("in_scope") is False):
            continue
        ref = label["terms"]
        ref_terms = NoteTerms.model_validate(ref)
        if acc not in markets:
            markets[acc] = build_market_inputs(ref_terms)
        payloads = {i: c["raw"] for i, c in sorted(samples.items()) if c["raw"] is not None}
        values = {i: {f: _hashable(field_value(p, f)) for f in CRITICAL_FIELDS} for i, p in payloads.items()}
        for i in sorted(samples):
            payload = payloads.get(i)
            rec = {"run_id": run_id, "accession": acc, "sample": i, "label_source": label["source"],
                   "output": payload is not None}
            if payload is None:
                rec.update(all_correct=False, all_correct_norm=False, priceable=False, hard=True,
                           delta=None, c_sample=0.0, material=True)
                out.append(rec)
                continue
            matches = _ev.field_matches(payload, ref)
            rec["matches"] = matches
            rec["all_correct"] = all(matches.values())
            rec["all_correct_norm"] = all(
                ok or (p == "autocall.levels_pct" and _ev.final_date_equivalent(payload, ref))
                for p, ok in matches.items())
            share = {f: sum(values[j][f] == values[i][f] for j in values) / len(values) for f in CRITICAL_FIELDS}
            rec["agreement"] = share
            rec["c_sample"] = min(share.values())
            check = _arms.evaluate(payload)
            rec["hard"] = (not check["schema_ok"]) or bool(check["hard"])
            rec["delta"], rec["priceable"] = None, False
            if check["schema_ok"]:
                ext, _ = to_note_terms(RawNoteTerms.model_validate(payload))
                try:
                    ext_market = None
                    if [u.name for u in ext.underlyings] != [u.name for u in ref_terms.underlyings]:
                        ext_market = build_market_inputs(ext)
                    pair = price_pair(ref_terms, ext, markets[acc], seed=args.seed, n_paths=args.n_paths,
                                      extracted_market=ext_market)
                    rec["delta"], rec["priceable"] = pair.signed_deviation, True
                except Exception as exc:  # noqa: BLE001 - unpriceable extraction
                    rec["price_error"] = f"{type(exc).__name__}: {exc}"
            rec["material"] = (not rec["priceable"]) or abs(rec["delta"]) >= MATERIAL_USD
            out.append(rec)
        print(f"  {run_id} {acc} {len(samples)} samples", flush=True)
    return out


def detection(recs: list[dict]) -> dict:
    mat = [r for r in recs if r["material"]]
    ok = [r for r in recs if not r["material"]]
    res = {"n_samples": len(recs), "n_material": len(mat),
           "material_rate": len(mat) / len(recs) if recs else None,
           "auroc_1_minus_c": auroc([1 - r["c_sample"] for r in mat], [1 - r["c_sample"] for r in ok])}
    for tau in (1.0, 0.8, 0.6):
        flag = lambda r: r["c_sample"] < tau  # noqa: E731
        res[f"tau_{tau}"] = {
            "agreement_detects": sum(flag(r) for r in mat),
            "agreement_or_rule_detects": sum(flag(r) or r["hard"] for r in mat),
            "false_alarm_rate": sum(flag(r) for r in ok) / len(ok) if ok else None,
        }
    res["rule_detects"] = sum(r["hard"] for r in mat)
    res["missed_at_tau_1"] = [
        {k: r.get(k) for k in ("run_id", "accession", "sample", "delta", "c_sample")}
        for r in mat if not (r["c_sample"] < 1.0 or r["hard"])
    ]
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--n-paths", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--out", default=str(RESULTS / "calibration.json"))
    args = parser.parse_args()

    labels = _ev.load_labels()
    markets: dict = {}
    recs: list[dict] = []
    for run_id in args.run_ids:
        recs.extend(score_run(run_id, labels, markets, args))

    replicates = []
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["run_id"], r["sample"])].append(r)
    for (run_id, i), g in sorted(groups.items()):
        priced = [abs(r["delta"]) for r in g if r["priceable"]]
        replicates.append({
            "run_id": run_id, "sample": i, "n": len(g),
            "em_pct": 100 * sum(r["all_correct"] for r in g) / len(g),
            "price_eq_pct": 100 * sum(r["all_correct_norm"] for r in g) / len(g),
            "unpriced": sum(not r["priceable"] for r in g),
            "max_abs_delta": max(priced) if priced else None,
            "n_material": sum(r["material"] for r in g),
        })

    by_run = {run_id: detection([r for r in recs if r["run_id"] == run_id]) for run_id in args.run_ids}
    out = {"material_usd": MATERIAL_USD, "overall": detection(recs), "by_run": by_run,
           "replicates": replicates, "samples": recs}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    print(f"\n{'run':24s} {'n':>4} {'mat':>4} {'AUROC':>6} {'det@1':>6} {'det@1+rule':>10} {'FA@1':>6}")
    for run_id, d in list(by_run.items()) + [("ALL", out["overall"])]:
        t = d["tau_1.0"]
        au = f"{d['auroc_1_minus_c']:.2f}" if d["auroc_1_minus_c"] is not None else "  -"
        fa = f"{t['false_alarm_rate']:.2f}" if t["false_alarm_rate"] is not None else "  -"
        print(f"{run_id:24s} {d['n_samples']:4d} {d['n_material']:4d} {au:>6} {t['agreement_detects']:6d} "
              f"{t['agreement_or_rule_detects']:10d} {fa:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
