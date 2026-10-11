"""Dataset statistics (table 1) and API cost / latency accounting (table 3).

Usage:
    python experiments/11_stats.py --runs stage2 stage2_nothink
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"

_spec = importlib.util.spec_from_file_location("ev", ROOT / "experiments" / "09_evaluate.py")
_ev = importlib.util.module_from_spec(_spec)
sys.modules["ev"] = _ev
_spec.loader.exec_module(_ev)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    labels = _ev.load_labels()
    cands = {
        c["accession"]: c
        for c in map(json.loads, (RESULTS / "candidates.jsonl").read_text().splitlines())
    }
    accs = sorted(a for a in labels if a in cands)
    terms = [labels[a]["terms"] for a in accs]
    prelabels = {
        a: json.loads((ROOT / "data/labels/prelabel" / f"{a}.json").read_text()) for a in accs
    }
    gold = {
        a: json.loads(p.read_text())
        for p in (ROOT / "data/labels/gold").glob("*.json")
        if (a := p.stem) in labels
    }

    n_obs = [len(t["observation_dates"]) for t in terms]
    n_und = [len(t["underlyings"]) for t in terms]
    tenor = [
        (int(t["maturity_date"][:4]) - int(t["pricing_date"][:4]))
        + (int(t["maturity_date"][5:7]) - int(t["pricing_date"][5:7])) / 12
        for t in terms
    ]
    derived = sum(
        any("derived" in (f.get("issue", "") + f.get("field", "")).lower() for f in pl.get("flags", []))
        or "derived" in json.dumps(pl.get("evidence", {}).get("payment_dates", {})).lower()
        for pl in prelabels.values()
    )
    stats = {
        "n_filings": len(accs),
        "n_gold": sum(labels[a]["source"] == "gold" for a in accs),
        "n_silver": sum(labels[a]["source"] == "silver" for a in accs),
        "issuers": dict(Counter(cands[a]["issuer_stratum"] for a in accs).most_common()),
        "filing_date_range": [min(cands[a]["filing_date"] for a in accs), max(cands[a]["filing_date"] for a in accs)],
        "single": sum(t["performance_type"] == "single" for t in terms),
        "worst_of": sum(t["performance_type"] == "worst_of" for t in terms),
        "n_underlyings_dist": dict(Counter(n_und)),
        "memory": sum(bool(t["coupon"]["memory"]) for t in terms),
        "step_down": sum(len(set(t["autocall"]["levels_pct"])) > 1 for t in terms),
        "strike_date_separate": sum(t.get("strike_date") is not None for t in terms),
        "payment_dates_derived": derived,
        "obs_mean": statistics.fmean(n_obs),
        "obs_min": min(n_obs),
        "obs_max": max(n_obs),
        "tenor_years_median": statistics.median(tenor),
        "tenor_years_range": [min(tenor), max(tenor)],
        "notional": dict(Counter(t["notional_per_unit"] for t in terms)),
        "final_date_callable": sum(
            len(t["autocall"]["levels_pct"]) == len(t["observation_dates"]) - t["autocall"]["first_observation_index"] + 1
            for t in terms
        ),
        "tokens_raw_mean": statistics.fmean(cands[a]["approx_tokens_raw"] for a in accs),
        "tokens_clean_mean": statistics.fmean(cands[a]["approx_tokens_clean"] for a in accs),
        "tokens_clean_median": statistics.median(cands[a]["approx_tokens_clean"] for a in accs),
        "tokens_clean_range": [min(cands[a]["approx_tokens_clean"] for a in accs), max(cands[a]["approx_tokens_clean"] for a in accs)],
        "tables_mean": statistics.fmean(cands[a]["n_tables"] for a in accs),
        "gold_verification_minutes": {a: g.get("provenance", {}).get("verification_minutes") for a, g in gold.items()},
        "gold_verification_minutes_range": {
            a: g.get("provenance", {}).get("verification_minutes_range")
            or [g.get("provenance", {}).get("verification_minutes")] * 2
            for a, g in gold.items()
        },
        "gold_verified_by": {a: g.get("provenance", {}).get("verified_by") for a, g in gold.items()},
        "gold_fields_changed": {a: g.get("provenance", {}).get("fields_changed") for a, g in gold.items()},
    }

    runs = {}
    for run in args.runs:
        calls = [json.loads(l) for l in (RESULTS / "runs" / run / "calls.jsonl").read_text().splitlines()]
        latest = {(c["accession"], c["kind"]): c for c in calls}
        cs = [c for c in latest.values() if c["accession"] in labels]
        ok = [c for c in cs if c.get("input_tokens")]
        inp = sum(c["input_tokens"] for c in ok)
        hit = sum(c.get("cached_input_tokens") or 0 for c in ok)
        vis = sum(c.get("output_tokens") or 0 for c in ok)
        think = sum(c.get("thinking_tokens") or 0 for c in ok)
        runs[run] = {
            "config": json.loads((RESULTS / "runs" / run / "config.json").read_text()),
            "n_calls": len(cs),
            "n_calls_all_attempts": len(calls),
            "api_failures": sum(c.get("error_kind") == "api" for c in cs),
            "api_retries": sum(c.get("api_retries") or 0 for c in cs),
            "schema_errors": sum(c.get("error_kind") == "schema" for c in cs),
            "input_tokens_mean": inp / len(ok),
            "cache_hit_share": hit / inp,
            "visible_output_mean": vis / len(ok),
            "thinking_mean": think / len(ok),
            "thinking_multiplier": think / vis if vis else None,
            "latency_mean_s": statistics.fmean(c["latency_s"] for c in ok),
            "latency_median_s": statistics.median(c["latency_s"] for c in ok),
            "total_cost_usd": sum(c.get("cost_usd", 0) for c in calls),
        }

    out = {"dataset": stats, "runs": runs}
    (RESULTS / "stats.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
