"""Build the candidate pool: stratified EDGAR search, screen, clean.

Writes ``experiments/results/candidates.jsonl`` with one record per surviving
filing and the cleaned text to ``data/cleaned/``.  Every download is cached under
``data/raw/``, so re-running does not re-hit EDGAR and the dataset is
reproducible.

Sampling (paper: Dataset): the query is issued once per issuer CIK, because
relevance-ordered full-text search otherwise returns almost only the two or
three issuers that file most.

Usage:
    python experiments/01_build_candidate_pool.py --contact you@example.edu \
        --start 2026-06-01 --end 2026-09-30 --per-issuer 8

    # Grow an existing pool: keep its records, scan up to 40 hits per issuer,
    # stop at 5 accepted per issuer, at most 2 per product series.
    python experiments/01_build_candidate_pool.py --contact you@example.edu \
        --append --per-issuer 40 --target-per-issuer 5 --series-cap 2

Exclusion counts are written to ``experiments/results/pool_exclusions.json``
for the paper's dataset section.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean import clean_filing
from src.crawl import EdgarClient, in_underlying_scope, is_preliminary, product_profile
from src.crawl.edgar import ISSUER_CIKS, EdgarError

CLEANED_DIR = ROOT / "data" / "cleaned"
RESULTS_DIR = ROOT / "experiments" / "results"

#: Both phrases must appear, which is what separates a Phoenix autocallable from
#: a plain autocallable with no contingent coupon.  The negative term excludes
#: preliminary term sheets at the index rather than after downloading them:
#: relevance ordering otherwise surfaces preliminary supplements first for
#: several issuers, and they are not usable ground truth.  Measured
#: effect on one issuer: 3,378 hits -> 1,683, with no final supplement lost.
QUERY = '"Autocallable" "Contingent Coupon" -"Subject to Completion"'

#: A term sheet runs tens of thousands of tokens, not hundreds of thousands; a
#: far larger document is a combined or base prospectus rather than a pricing
#: supplement for a single note.
MAX_CLEAN_TOKENS = 80_000
MIN_CLEAN_TOKENS = 3_000

#: Observation schedules and call-level ladders live in tables.  A pricing
#: supplement with almost none is not the document this pipeline expects.
MIN_TABLES = 5


def series_key(issuer: str, underlyings: list[str], profile: dict) -> tuple:
    """Issuer, basket and coupon style: filings sharing all three are one series.

    Same-series supplements are near-identical templates with different dates
    and levels, so capping them keeps accuracy from being inflated.
    """
    return (issuer, tuple(sorted(underlyings)), profile["worst_of"], profile["memory"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", required=True, help="contact email for the SEC User-Agent")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 6, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    parser.add_argument("--per-issuer", type=int, default=8)
    parser.add_argument(
        "--issuers", nargs="*", default=None,
        help="subset of issuer names from ISSUER_CIKS; default is all",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="keep the existing candidates.jsonl records and add new filings after them",
    )
    parser.add_argument(
        "--target-per-issuer", type=int, default=None,
        help="stop accepting an issuer's filings once it has this many (existing included)",
    )
    parser.add_argument(
        "--series-cap", type=int, default=None,
        help="at most this many filings per product series",
    )
    args = parser.parse_args()

    issuers = ISSUER_CIKS
    if args.issuers:
        missing = [n for n in args.issuers if n not in ISSUER_CIKS]
        if missing:
            parser.error(f"unknown issuer(s): {missing}; known: {sorted(ISSUER_CIKS)}")
        issuers = {n: ISSUER_CIKS[n] for n in args.issuers}

    client = EdgarClient(args.contact)
    print(f"searching 424B2 {QUERY}  {args.start}..{args.end}", flush=True)
    print(f"  {len(issuers)} issuer strata x {args.per_issuer} filings\n", flush=True)

    by_issuer = client.search_by_issuer(
        QUERY, start=args.start, end=args.end,
        per_issuer=args.per_issuer, issuers=issuers,
    )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = RESULTS_DIR / "candidates.jsonl"
    existing: list[dict] = []
    if args.append and out.exists():
        existing = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    seen = {r["accession"] for r in existing}
    per_issuer_count: Counter[str] = Counter(r["issuer_stratum"] for r in existing)
    per_series: Counter[tuple] = Counter()
    for r in existing:
        # Records from before profiling existed are left untouched on disk;
        # their series is recomputed from the cleaned text for the cap only.
        prof = r.get("profile")
        if prof is None:
            text = (CLEANED_DIR / f"{r['accession']}_{r['document']}.txt").read_text(encoding="utf-8")
            prof = product_profile(text, matched_underlyings=r["matched_underlyings"]).to_dict()
        per_series[series_key(r["issuer_stratum"], r["matched_underlyings"], prof)] += 1

    kept: list[dict] = []
    skipped: Counter[str] = Counter()
    pdf_recovered = 0

    for issuer, refs in by_issuer.items():
        if not refs:
            skipped["no filings in window"] += 1
            print(f"{issuer[:40]:42s} (none in window)", flush=True)
            continue
        print(f"{issuer[:40]:42s}", flush=True)

        for ref in refs:
            if args.target_per_issuer and per_issuer_count[issuer] >= args.target_per_issuer:
                break
            if ref.accession in seen:
                continue
            seen.add(ref.accession)
            try:
                was_pdf = ref.is_pdf
                ref = client.resolve_html(ref)
                if was_pdf:
                    pdf_recovered += 1
                html = client.download(ref)
            except EdgarError as exc:
                skipped[f"fetch failed: {exc.__class__.__name__}"] += 1
                continue

            doc = clean_filing(html)

            if doc.n_tables < MIN_TABLES:
                skipped[f"fewer than {MIN_TABLES} tables"] += 1
                continue
            if not MIN_CLEAN_TOKENS <= doc.approx_tokens_clean <= MAX_CLEAN_TOKENS:
                skipped["token count out of range"] += 1
                continue

            verdict = is_preliminary(doc.text)
            if verdict.is_preliminary:
                skipped["preliminary term sheet"] += 1
                continue

            scope = in_underlying_scope(doc.text)
            if not scope.in_scope:
                skipped[f"out of underlying scope ({scope.reasons[0]})"] += 1
                continue

            profile = product_profile(doc.text, matched_underlyings=scope.matched_underlyings)
            if profile.unregistered_index:
                skipped["out of underlying scope (unregistered index in basket)"] += 1
                continue
            if profile.buffered:
                skipped["out of template scope (buffered downside)"] += 1
                continue
            if profile.daily_monitoring:
                skipped["out of template scope (daily barrier monitoring)"] += 1
                continue

            key = series_key(issuer, scope.matched_underlyings, profile.to_dict())
            if args.series_cap and per_series[key] >= args.series_cap:
                skipped["product-series cap reached"] += 1
                continue
            per_series[key] += 1
            per_issuer_count[issuer] += 1

            (CLEANED_DIR / f"{ref.filing_id}.txt").write_text(doc.text, encoding="utf-8")
            kept.append(
                ref.to_dict()
                | {
                    "issuer_stratum": issuer,
                    "profile": profile.to_dict(),
                    "recovered_from_pdf": was_pdf,
                    "matched_underlyings": scope.matched_underlyings,
                    "n_tables": doc.n_tables,
                    "chars_raw": doc.chars_raw,
                    "chars_clean": doc.chars_clean,
                    "approx_tokens_raw": doc.approx_tokens_raw,
                    "approx_tokens_clean": doc.approx_tokens_clean,
                    "reduction_pct": round(doc.reduction_pct, 1),
                    "dropped_sections": [d.heading for d in doc.dropped_sections],
                    "running_headers_removed": doc.running_headers_removed,
                }
            )
            print(
                f"    {ref.filing_date}  {doc.approx_tokens_raw:>6,}->"
                f"{doc.approx_tokens_clean:>6,} tok (-{doc.reduction_pct:>4.1f}%)  "
                f"{doc.n_tables:>3} tables{'  [from pdf]' if was_pdf else ''}",
                flush=True,
            )

    with out.open("w", encoding="utf-8") as handle:
        for record in existing + kept:
            handle.write(json.dumps(record) + "\n")

    exclusions = {
        "run": {
            "query": QUERY,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "per_issuer_scanned": args.per_issuer,
            "target_per_issuer": args.target_per_issuer,
            "series_cap": args.series_cap,
            "appended_to": len(existing),
        },
        "kept_new": len(kept),
        "total_in_pool": len(existing) + len(kept),
        "recovered_from_pdf": pdf_recovered,
        "excluded": dict(skipped.most_common()),
    }
    (RESULTS_DIR / "pool_exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")

    print(f"\n{'=' * 72}\nkept {len(kept)} new filings ({len(existing)} existing) -> {out.relative_to(ROOT)}")
    print(f"recovered from PDF-only search hits: {pdf_recovered}")
    if skipped:
        print("skipped:")
        for reason, count in skipped.most_common():
            print(f"  {count:>4}  {reason}")
    if kept:
        tokens = sorted(r["approx_tokens_clean"] for r in kept)
        reductions = [r["reduction_pct"] for r in kept]
        print(
            f"cleaned tokens  min {tokens[0]:,}  median {tokens[len(tokens) // 2]:,}  "
            f"max {tokens[-1]:,}"
        )
        print(
            f"cleaning reduction  min {min(reductions):.1f}%  "
            f"median {sorted(reductions)[len(reductions) // 2]:.1f}%  "
            f"max {max(reductions):.1f}%"
        )
        print("per issuer:")
        for issuer, count in Counter(r["issuer_stratum"] for r in existing + kept).most_common():
            print(f"  {count:>3}  {issuer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
