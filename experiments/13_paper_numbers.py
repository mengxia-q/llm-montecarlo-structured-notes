"""Write every number the paper quotes to ``paper/numbers.tex`` as LaTeX macros.

The manuscript never types a result by hand: rerun this after any change to the
labels or runs and recompile.  Run 09, 10 and 11 first.

Usage:
    python experiments/13_paper_numbers.py --think stage2 --nothink stage2_nothink
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "numbers.tex"

ARMS = ("B1", "B2", "B3", "Ours")
ARM_TAG = {"B1": "Bone", "B2": "Btwo", "B3": "Bthree", "Ours": "Ours"}
RUN_TAG = {"think": "On", "nothink": "Off"}
RELEASE_MONTH = "2026-09-01"  # DeepSeek-V4.1-Flash release month; no training cutoff is published


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def money(x: float | None, d: int = 2) -> str:
    return "--" if x is None else f"{x:,.{d}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--think", default="stage2")
    parser.add_argument("--nothink", default="stage2_nothink")
    args = parser.parse_args()
    runs = {"think": args.think, "nothink": args.nothink}

    macros: dict[str, str] = {}
    stats = json.loads((RES / "stats.json").read_text())
    ds = stats["dataset"]
    macros.update(
        NFilings=str(ds["n_filings"]),
        NGold=str(ds["n_gold"]),
        NSilver=str(ds["n_silver"]),
        NIssuers=str(len(ds["issuers"])),
        NSingle=str(ds["single"]),
        NWorstOf=str(ds["worst_of"]),
        NMemory=str(ds["memory"]),
        NStrike=str(ds["strike_date_separate"]),
        NDerived=str(ds["payment_dates_derived"]),
        NFinalCallable=str(ds["final_date_callable"]),
        ObsMean=f"{ds['obs_mean']:.1f}",
        ObsMin=str(ds["obs_min"]),
        ObsMax=str(ds["obs_max"]),
        TenorMin=f"{ds['tenor_years_range'][0]:.0f}",
        TenorMax=f"{ds['tenor_years_range'][1]:.0f}",
        TenorMedian=f"{ds['tenor_years_median']:.0f}",
        TokMean=f"{ds['tokens_clean_mean'] / 1000:.1f}k",
        TokMin=f"{ds['tokens_clean_range'][0] / 1000:.1f}k",
        TokMax=f"{ds['tokens_clean_range'][1] / 1000:.1f}k",
        TablesMean=f"{ds['tables_mean']:.0f}",
        DateFrom=ds["filing_date_range"][0],
        DateTo=ds["filing_date_range"][1],
        NThreeUnd=str(ds["n_underlyings_dist"].get("3", 0)),
        NTwoUnd=str(ds["n_underlyings_dist"].get("2", 0)),
    )
    # Per-file times where recorded, else the reported range's end points.
    mins = [m for r in ds["gold_verification_minutes_range"].values() for m in r if m]
    by = list(ds["gold_verified_by"].values())
    macros["NGoldAuthor"] = str(sum(v == "author" for v in by))
    macros["NGoldSecond"] = str(sum(v != "author" for v in by))
    changed = [len(v or []) for v in ds["gold_fields_changed"].values() if v is not None]
    macros.update(
        HumanMinutesMin=str(min(mins)) if mins else "--",
        HumanMinutesMax=str(max(mins)) if mins else "--",
        SilverFieldsChanged=str(sum(changed)),
        SilverFilesChanged=str(sum(c > 0 for c in changed)),
        SilverFieldsChangedPhrase="no field" if sum(changed) == 0 else f"{sum(changed)} fields",
    )

    filed = {
        c["accession"]: c["filing_date"]
        for c in map(json.loads, (RES / "candidates.jsonl").read_text().splitlines())
    }
    for key, run in runs.items():
        t = RUN_TAG[key]
        r = stats["runs"][run]
        macros[f"Lat{t}"] = f"{r['latency_median_s']:.1f}"
        macros[f"LatMean{t}"] = f"{r['latency_mean_s']:.1f}"
        macros[f"In{t}"] = f"{r['input_tokens_mean'] / 1000:.1f}k"
        macros[f"Cache{t}"] = pct(r["cache_hit_share"], 0)
        macros[f"Vis{t}"] = f"{r['visible_output_mean']:.0f}"
        macros[f"Think{t}"] = f"{r['thinking_mean']:.0f}"
        macros[f"ThinkMult{t}"] = f"{r['thinking_multiplier']:.1f}" if r["thinking_multiplier"] else "0"
        macros[f"Cost{t}"] = money(r["total_cost_usd"])
        macros[f"NCalls{t}"] = str(r["n_calls"])
        macros[f"ApiFail{t}"] = str(r["api_failures"])
        summary = json.loads((RES / "runs" / run / "summary.json").read_text())
        rows = [json.loads(line) for line in (RES / "runs" / run / "eval_rows.jsonl").read_text().splitlines()]
        # Contamination check: B1 exact match on filings filed from the model's release month on.
        b1 = [r for r in rows if r["arm"] == "B1"]
        late = [r for r in b1 if filed[r["accession"]] >= RELEASE_MONTH]
        early = [r for r in b1 if filed[r["accession"]] < RELEASE_MONTH]
        macros["NLate"], macros["NEarly"] = str(len(late)), str(len(early))
        macros[f"LateBone{t}"] = str(sum(r["all_correct"] for r in late))
        macros[f"EarlyBone{t}"] = str(sum(r["all_correct"] for r in early))
        for subset in ("all", "gold"):
            sub = "G" if subset == "gold" else ""
            for arm in ARMS:
                s = summary.get(subset, {}).get(arm)
                if not s:
                    continue
                p = f"{ARM_TAG[arm]}{t}{sub}"
                macros[f"Field{p}"] = pct(s["field_accuracy"])
                macros[f"File{p}"] = pct(s["file_all_correct"])
                macros[f"FileN{p}"] = pct(s["file_all_correct_norm"])
                macros[f"Auto{p}"] = pct(s["per_category_accuracy"]["autocall terms"])
                # Files (not fields) with an autocall error: the category pools two fields.
                macros[f"AutoFiles{p}"] = str(sum(
                    not all(ok for f, ok in r["matches"].items() if f.startswith("autocall."))
                    for r in rows
                    if r["arm"] == arm and (subset == "all" or r["label_source"] == "gold")
                ))
                macros[f"Coup{p}"] = pct(s["per_category_accuracy"]["coupon terms"])
                macros[f"Unpr{p}"] = pct(s["unpriceable_rate"])
                macros[f"DMax{p}"] = money(s["delta_abs_max"])
                macros[f"DMean{p}"] = money(s["delta_abs_mean"], 3)
                macros[f"DNinety{p}"] = money(s["delta_abs_p90"], 3)
                macros[f"GtOne{p}"] = pct(s["share_delta_gt_1"])
                macros[f"CostFile{p}"] = f"{1000 * s['mean_cost_usd']:.1f}"
                macros[f"Calls{p}"] = f"{s['mean_calls']:.2f}"
                macros[f"N{p}"] = str(s["n"])
                if "coverage_curve" in s:
                    full = [c for c in s["coverage_curve"] if c["selective_accuracy"] == 1.0]
                    best = max(full, key=lambda c: c["coverage"]) if full else None
                    macros[f"CovAtOne{p}"] = pct(best["coverage"], 0) if best else "0"
                    macros[f"TauAtOne{p}"] = f"{best['tau']:.2f}" if best else "--"
                    full_n = [c for c in s["coverage_curve"] if c["selective_accuracy_norm"] == 1.0]
                    best_n = max(full_n, key=lambda c: c["coverage"]) if full_n else None
                    macros[f"CovAtOneN{p}"] = pct(best_n["coverage"], 0) if best_n else "0"
                    c80 = next(c for c in s["coverage_curve"] if abs(c["tau"] - 0.8) < 1e-9)
                    macros[f"CovEighty{p}"] = pct(c80["coverage"], 0)
                    macros[f"AccEighty{p}"] = pct(c80["selective_accuracy"] or 0, 0)

    sens = json.loads((RES / "runs" / args.nothink / "sensitivity.json").read_text())
    rank = {r["field"]: r for r in sens["ranking"]}
    for field, tag in {
        "coupon.rate_per_period_pct": "Unit",
        "coupon.barrier_pct": "CBar",
        "autocall.levels_pct": "ALev",
        "knock_in.monitoring": "KMon",
        "knock_in.barrier_pct": "KBar",
        "autocall.first_observation_index": "AFirst",
        "coupon.memory": "Mem",
        "underlyings.initial_level": "Init",
        "observation_dates": "Drop",
    }.items():
        if field in rank:
            macros[f"Sens{tag}"] = money(rank[field]["median_abs"])
            macros[f"Sens{tag}Max"] = money(rank[field]["max_abs"])
    stab = [s for s in sens["stability"] if abs(s["delta_seed0"]) > 1]
    if stab:
        macros["StabDelta"] = money(abs(stab[0]["delta_seed0"]))
        macros["StabStd"] = money(stab[0]["std"], 2)
    small = [s for s in sens["stability"] if 0 < abs(s["delta_seed0"]) < 1]
    if small:
        macros["StabSmallStd"] = f"{max(s['std'] for s in small):.5f}"

    conv = json.loads((RES / "convergence.json").read_text())
    n_prod = conv["config"]["production_n"]
    se_sobol = [next(x for x in note["methods"]["sobol"] if x["n_paths"] == n_prod)["std"]
                for note in conv["notes"].values()]
    macros.update(
        SobolSEMin=f"{min(se_sobol):.2f}",
        SobolSEMax=f"{max(se_sobol):.2f}",
        NConvNotes=str(len(conv["notes"])),
    )

    r2_path = ROOT / "data" / "labels" / "round2" / "round2_report.json"
    if r2_path.exists():
        r2 = json.loads(r2_path.read_text())
        seeded = r2["seeded"]
        macros["SeededN"] = str(len(seeded))
        macros["SeededFound"] = str(sum(bool(s["detected"]) for s in seeded))
        macros["OverlapN"] = str(len(r2["overlap"]))
        macros["OverlapAgree"] = str(sum(o["critical_agree"] for o in r2["overlap"]))
        macros["OverlapTotal"] = str(sum(o["critical_total"] for o in r2["overlap"]))
        macros["OverlapFullAgree"] = str(sum(not o["all_fields_disagree"] for o in r2["overlap"]))
        for v, tag in (("author", "Author"), ("second", "Second")):
            t = r2["verifiers"][v]
            macros[f"Min{tag}Mean"] = f"{t['minutes_mean']:.1f}"
            macros[f"Min{tag}Median"] = f"{t['minutes_median']:.0f}"
            macros[f"Min{tag}Lo"] = f"{t['minutes_range'][0]:.0f}"
            macros[f"Min{tag}Hi"] = f"{t['minutes_range'][1]:.0f}"
            macros[f"NFiles{tag}Rtwo"] = str(len(t["files"]))
        macros["GenuineCorrectionsRtwo"] = str(sum(len(f["corrections_to_prelabel"]) for t in r2["verifiers"].values() for f in t["files"]))

    lines = ["% Generated by experiments/13_paper_numbers.py -- do not edit by hand."]
    for name, value in sorted(macros.items()):
        safe = value.replace("%", r"\%").replace("$", r"\$")
        lines.append(f"\\newcommand{{\\{name}}}{{{safe}}}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
