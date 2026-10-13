"""Stress-test input: the 50 filings cleaned without rowspan expansion (revision, reviewer point 5).

The paper's cleaner repeats a ``rowspan`` cell into every row it spans; without
that, the initial level of a second underlying slides under the coupon column
(Method, Cleaning).  Extracting from this degraded text produces genuine,
price-relevant model errors on which vote agreement can be calibrated, rather
than errors that have already been removed by the vote.

The script first checks that cleaning ``data/raw`` with the default settings
reproduces ``data/cleaned`` byte for byte, so the only difference in the output
is the rowspan handling.

Usage:
    python experiments/20_clean_norowspan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean import clean_filing

CLEANED = ROOT / "data" / "cleaned"
OUT = ROOT / "data" / "cleaned_norowspan"


def main() -> int:
    arms = ROOT / "experiments" / "results" / "runs" / "stage2" / "arms.jsonl"
    accessions = sorted({json.loads(l)["accession"] for l in arms.read_text().splitlines()})
    OUT.mkdir(parents=True, exist_ok=True)
    changed = 0
    for acc in accessions:
        (clean_path,) = CLEANED.glob(f"{acc}_*.txt")
        html = (ROOT / "data" / "raw" / clean_path.name.removesuffix(".txt")).read_text(errors="replace")
        if clean_filing(html).text != clean_path.read_text():
            print(f"{acc}: default cleaning does not reproduce data/cleaned", file=sys.stderr)
            return 1
        text = clean_filing(html, expand_rowspan=False).text
        (OUT / clean_path.name).write_text(text)
        changed += text != clean_path.read_text()
    print(f"{len(accessions)} filings written to {OUT.relative_to(ROOT)}; {changed} differ from data/cleaned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
