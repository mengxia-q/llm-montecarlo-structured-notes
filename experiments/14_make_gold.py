"""Promote a human-verified prelabel to a gold label, then refresh every paper number.

Corrections are given as dotted paths with JSON values; the prelabel itself is
never modified, so silver-vs-gold differences stay measurable.

Usage:
    python experiments/14_make_gold.py 0001214659-26-011449 --minutes 15
    python experiments/14_make_gold.py 0001918704-26-018771 --minutes 20 \
        --set coupon.rate_per_period_pct=1.75 --set 'payment_dates=["2026-10-06", ...]'
    python experiments/14_make_gold.py --refresh      # only rerun evaluation + numbers
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels"
PY = sys.executable


def set_path(obj: dict, path: str, value) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else obj[part]
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        obj[last] = value


def refresh() -> None:
    for script, extra in (
        ("09_evaluate.py", ["--run-id", "stage2"]),
        ("09_evaluate.py", ["--run-id", "stage2_nothink"]),
        ("11_stats.py", ["--runs", "stage2", "stage2_nothink"]),
        ("13_paper_numbers.py", []),
    ):
        subprocess.run([PY, str(ROOT / "experiments" / script), *extra], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("refreshed evaluation, stats and paper/numbers.tex")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accession", nargs="?")
    parser.add_argument("--minutes", type=float)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=JSON")
    parser.add_argument("--notes", default="")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.accession:
        if args.minutes is None:
            parser.error("--minutes is required (human-cost data for the cost table)")
        pre = json.loads((LABELS / "prelabel" / f"{args.accession}.json").read_text())
        terms = copy.deepcopy(pre["terms"])
        changed = []
        for item in args.set:
            path, _, raw = item.partition("=")
            set_path(terms, path, json.loads(raw))
            changed.append(path)
        gold = {
            "accession": args.accession,
            "source_file": pre["source_file"],
            "terms": terms,
            "provenance": {
                "method": "pre-annotation verified field by field against the filing",
                "prelabel_file": f"data/labels/prelabel/{args.accession}.json",
                "prelabel_annotator": pre.get("annotator"),
                "verified_by": "author",
                "verified_on": date.today().isoformat(),
                "fields_changed": changed,
                "verification_minutes": args.minutes,
                "notes": args.notes,
            },
        }
        out = LABELS / "gold" / f"{args.accession}.json"
        out.write_text(json.dumps(gold, indent=2) + "\n")
        print(f"wrote {out.relative_to(ROOT)} ({len(changed)} field(s) changed)")
        check = subprocess.run([PY, str(ROOT / "experiments" / "06_check_prelabels.py"), str(out)],
                               capture_output=True, text=True)
        print(check.stdout.strip().splitlines()[-1] if check.stdout else check.stderr[-300:])
    if args.accession or args.refresh:
        refresh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
