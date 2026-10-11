"""Mechanical checks on pre-annotations before a human verifies them.

A pre-annotation in ``data/labels/prelabel/<accession>.json`` carries the
extracted ``terms``, a verbatim ``evidence`` quote with line numbers for every
field, reviewer ``flags`` and a ``scope`` verdict.  This script checks, per file:

* every evidence quote occurs verbatim (up to whitespace) in the cleaned filing,
  and within its stated line span, so the human verifier can trust the pointer;
* the terms convert to a priceable ``NoteTerms``;
* the business rules (``src/validate/rules.py``), including the initial-level cross-check
  against the close on ``initial_level_date`` (strike date if there is one).

It repairs nothing.  A non-zero exit status means at least one file has a
problem the verifier should know about before starting.

Usage:
    python experiments/06_check_prelabels.py                 # every pre-annotation
    python experiments/06_check_prelabels.py data/labels/prelabel/<accession>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract.llm_schema import RawNoteTerms, to_note_terms
from src.market import MarketDataError, UnknownUnderlyingError, price_history, resolve_underlying
from src.market.data import _close_on_or_before
from src.validate import validate_terms

PRELABEL_DIR = ROOT / "data" / "labels" / "prelabel"

#: Lines of slack either side of a stated evidence span.
SPAN_SLACK = 2


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_evidence(record: dict) -> tuple[list[str], list[str]]:
    """Quotes not found in the filing, and quotes found outside their line span."""
    source = (ROOT / record["source_file"]).read_text(encoding="utf-8")
    lines = source.splitlines()
    whole = _norm(source)
    missing, misplaced = [], []
    for key, evidence in record["evidence"].items():
        quote = _norm(evidence.get("quote") or "")
        if not quote or quote not in whole:
            missing.append(key)
            continue
        lo, hi = evidence.get("lines") or (None, None)
        if lo and hi:
            span = _norm("\n".join(lines[max(0, lo - 1 - SPAN_SLACK): hi + SPAN_SLACK]))
            if quote not in span:
                misplaced.append(f"{key} (lines {lo}-{hi})")
    return missing, misplaced


def check_file(path: Path) -> bool:
    record = json.loads(path.read_text(encoding="utf-8"))
    print("=" * 78)
    print(path.relative_to(ROOT))
    ok = True

    missing, misplaced = check_evidence(record)
    print(f"  evidence: {len(record['evidence'])} fields")
    if missing:
        ok = False
        print(f"    ! quote not found verbatim: {', '.join(missing)}")
    if misplaced:
        ok = False
        print(f"    ! quote outside its line span: {', '.join(misplaced)}")

    terms, violations = to_note_terms(RawNoteTerms.model_validate(record["terms"]))
    if terms is None:
        print("  schema: FAILED")
        for violation in violations:
            print(f"    ! {violation}")
        return False
    print("  schema: OK")

    closes: dict[str, float] = {}
    day = terms.initial_level_date
    for underlying in terms.underlyings:
        try:
            ticker = resolve_underlying(underlying.name, underlying.ticker).yahoo_ticker
            series = price_history(ticker, day - timedelta(days=10), day)
            close, on = _close_on_or_before(series, day, ticker)
        except (UnknownUnderlyingError, MarketDataError) as exc:
            print(f"  initial level: {underlying.name}: no close ({exc})")
            continue
        closes[ticker] = close
        rel = 100 * (underlying.initial_level / close - 1)
        print(
            f"  initial level: {ticker} filed {underlying.initial_level:,.2f} "
            f"vs close {close:,.2f} on {on} ({rel:+.3f}%)"
        )

    report = validate_terms(terms, initial_level_closes=closes)
    if report.findings:
        ok = ok and report.ok
        for finding in report.findings:
            print(f"  rule: {finding}")
    else:
        print("  rules: clean")

    flags = record.get("flags", [])
    scope = record.get("scope") or {}
    print(f"  flags for the verifier: {len(flags)}   in scope: {scope.get('in_scope')}")
    for flag in flags:
        print(f"    - {flag['field']}: {flag['issue'][:110]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    files = [p.resolve() for p in args.files] or sorted(PRELABEL_DIR.glob("*.json"))
    if not files:
        print(f"no pre-annotations under {PRELABEL_DIR}", file=sys.stderr)
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = [check_file(p) for p in files]

    print("=" * 78)
    print(f"{sum(results)}/{len(results)} pre-annotations pass every check")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
