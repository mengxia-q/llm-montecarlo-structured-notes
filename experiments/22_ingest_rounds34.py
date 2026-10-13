"""Ingest rounds three and four (revision, 2026-10-13) into gold labels.

Round three: 5 filings annotated from scratch by one person each, on the
annotator's own machine, then transcribed by the corresponding author into
``data/labels/round3/``.  Each annotation is compared field by field with the
prelabel on the evaluation's terms (critical fields, ``PCT_TOLERANCE``); payment
dates left empty because the filing only gives a rule are derived with the NYSE
calendar, as the guideline prescribes, before comparison.  Disagreements are
listed for adjudication; with none, the gold terms equal the prelabel.

Round four: the remaining 22 silver labels verified field by field from the
prelabel; the packets are diffed against ``_original/``.

Writes ``data/labels/gold/<accession>.json`` for every file of both rounds and
``data/labels/round34_report.json``.

Usage:
    python experiments/22_ingest_rounds34.py          # report only
    python experiments/22_ingest_rounds34.py --write
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract.vote import field_value
from src.schema import CRITICAL_FIELDS

LABELS = ROOT / "data" / "labels"
PY = sys.executable
DATE = "2026-10-13"

_spec = importlib.util.spec_from_file_location("ev", ROOT / "experiments" / "09_evaluate.py")
_ev = importlib.util.module_from_spec(_spec)
sys.modules["ev"] = _ev
_spec.loader.exec_module(_ev)

#: Payment-date rules quoted in the round-three flags: business days after each
#: observation date, final payment on the maturity date.
RULE_DAYS = {"0000950103-26-010350": 3, "0001193125-26-407802": 5}

#: Role labels written to the released labels and report, as in round two
#: ("author", "second").  The name -> role mapping is kept in
#: data/labels/roles.json, which is not released.
ROLE = json.loads((LABELS / "roles.json").read_text(encoding="utf-8"))


def timing(path: Path, who: str) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["accession"]: {"person": ROLE[r[who].strip()], "minutes": float(r["minutes"])} for r in csv.DictReader(fh)}


def derive_payments(terms: dict, plus: int) -> list[str]:
    out = subprocess.run([PY, "-m", "src.market.calendar", *terms["observation_dates"][:-1], "--plus", str(plus)],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return out + [terms["maturity_date"]]


def compare(human: dict, pre: dict) -> dict[str, dict]:
    diffs = {}
    for path, ok in _ev.field_matches(human, pre).items():
        if not ok:
            diffs[path] = {"human": field_value(human, path), "prelabel": field_value(pre, path)}
    for key in ("payment_dates", "pricing_date", "strike_date", "issue_date", "notional_per_unit",
                "performance_type", "downside_participation_pct", "estimated_value_per_unit"):
        if human.get(key) != pre.get(key):
            diffs[key] = {"human": human.get(key), "prelabel": pre.get(key)}
    if human["knock_in"]["monitoring"] != pre["knock_in"]["monitoring"]:
        diffs["knock_in.monitoring"] = {"human": human["knock_in"]["monitoring"], "prelabel": pre["knock_in"]["monitoring"]}
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    report: dict = {"round3": {}, "round4": {}}
    gold: dict[str, dict] = {}

    t3 = timing(LABELS / "round3" / "timing.csv", "annotator")
    for acc, t in sorted(t3.items()):
        human = json.loads((LABELS / "round3" / f"{acc}.json").read_text(encoding="utf-8"))["terms"]
        pre_rec = json.loads((LABELS / "prelabel" / f"{acc}.json").read_text())
        pre = pre_rec["terms"]
        derived = False
        if not human["payment_dates"] and acc in RULE_DAYS:
            human["payment_dates"] = derive_payments(human, RULE_DAYS[acc])
            derived = True
        diffs = compare(human, pre)
        n_crit_ok = sum(_ev.field_matches(human, pre).values())
        report["round3"][acc] = {"annotator": t["person"], "minutes": t["minutes"], "payment_dates_derived": derived,
                                 "critical_agree": f"{n_crit_ok}/{len(CRITICAL_FIELDS)}", "disagreements": diffs}
        if not diffs:
            gold[acc] = {"terms": pre, "pre": pre_rec, "provenance": {
                "method": "independent annotation from scratch (no prelabel shown), compared field by field "
                          "with the prelabel; no disagreement, so gold = prelabel",
                "annotated_by": t["person"], "annotator_is_author": not t["person"].startswith("verifier"),
                "annotation_minutes": t["minutes"], "round": 3,
                "transcribed_by": "corresponding author (annotation done on the annotator's own machine)",
                "payment_dates_derived_by_program": derived}}

    t4 = timing(LABELS / "round4" / "timing.csv", "verifier")
    for acc, t in sorted(t4.items()):
        packet = json.loads((LABELS / "round4" / f"{acc}.json").read_text(encoding="utf-8"))
        orig = json.loads((LABELS / "round4" / "_original" / f"{acc}.json").read_text(encoding="utf-8"))
        changed = [k for k in orig["terms"] if packet["terms"].get(k) != orig["terms"].get(k)]
        pre_rec = json.loads((LABELS / "prelabel" / f"{acc}.json").read_text())
        report["round4"][acc] = {"verifier": t["person"], "minutes": t["minutes"], "fields_changed": changed}
        gold[acc] = {"terms": packet["terms"], "pre": pre_rec, "provenance": {
            "method": "pre-annotation verified field by field against the filing (round 4, timed, no seeded errors)",
            "verified_by": t["person"], "verifier_is_author": not t["person"].startswith("verifier"),
            "verification_minutes": t["minutes"], "round": 4, "fields_changed": changed}}

    r3 = report["round3"].values()
    r4 = report["round4"].values()
    report["summary"] = {
        "round3_files": len(r3), "round3_files_full_agreement": sum(not r["disagreements"] for r in r3),
        "round3_mean_minutes": sum(r["minutes"] for r in r3) / len(r3),
        "round4_files": len(r4), "round4_fields_changed": sum(len(r["fields_changed"]) for r in r4),
        "round4_mean_minutes": sum(r["minutes"] for r in r4) / len(r4),
        "people": sorted({r["annotator"] for r in r3} | {r["verifier"] for r in r4}),
    }
    print(json.dumps(report["summary"], indent=2))
    for acc, r in report["round3"].items():
        if r["disagreements"]:
            print(f"round3 {acc}: {r['disagreements']}")

    if args.write:
        (LABELS / "round34_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        for acc, g in gold.items():
            prov = {"prelabel_file": f"data/labels/prelabel/{acc}.json",
                    "prelabel_annotator": g["pre"].get("annotator"), "verified_on": DATE, **g["provenance"]}
            rec = {"accession": acc, "source_file": g["pre"]["source_file"], "terms": g["terms"], "provenance": prov}
            (LABELS / "gold" / f"{acc}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {len(gold)} gold labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
