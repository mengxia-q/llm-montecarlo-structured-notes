"""End-to-end vertical slice: filing -> extraction -> validation -> price -> Delta.

Runs the whole pipeline on a handful of real filings to prove the chain connects
and to measure per-field price sensitivity.

Pilot, written before any labels existed: no paper number comes from this
script.  The price deviation is demonstrated by *controlled perturbation*: each
critical field is perturbed one at a time with a realistic extraction error, and
the resulting Delta is measured against the unperturbed parameter set with
shared random numbers.  The perturbations defined here are reused by
``10_sensitivity.py``, which produces figure 3 over every labelled filing.

Usage:
    python experiments/02_vertical_slice.py --n-filings 3 --k 3
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.getLogger("google_genai").setLevel(logging.ERROR)

from src.eval import price_pair
from src.extract import build_prompt, get_provider, to_note_terms
from src.extract.vote import majority_payload, vote_fields
from src.market import MarketDataError, build_market_inputs, initial_level_closes
from src.market.tickers import UnknownUnderlyingError
from src.pricing.engine import UnsupportedProductError
from src.schema import NoteTerms
from src.validate import validate_terms

RESULTS = ROOT / "experiments" / "results"
CLEANED = ROOT / "data" / "cleaned"


@dataclass
class Perturbation:
    """One realistic extraction error, applied to a parameter dump."""

    name: str
    description: str
    apply: object  # Callable[[dict], None]


def _perturbations() -> list[Perturbation]:
    def coupon_barrier_up(d: dict) -> None:
        d["coupon"]["barrier_pct"] += 5.0

    def knock_in_down(d: dict) -> None:
        d["knock_in"]["barrier_pct"] = max(1.0, d["knock_in"]["barrier_pct"] - 5.0)

    def coupon_rate_per_annum(d: dict) -> None:
        # The single most common unit error: copying the quoted per-annum rate
        # into a per-period field.  Payments per year from the schedule itself,
        # since monthly and quarterly notes are both common.
        from datetime import date as _date

        obs = [_date.fromisoformat(x) for x in d["observation_dates"]]
        gaps = sorted((b - a).days for a, b in zip(obs, obs[1:])) or [91]
        n_per_year = max(1, round(365.25 / gaps[len(gaps) // 2]))
        d["coupon"]["rate_per_period_pct"] *= n_per_year

    def memory_flipped(d: dict) -> None:
        d["coupon"]["memory"] = not d["coupon"]["memory"]

    def autocall_starts_one_early(d: dict) -> None:
        idx = d["autocall"]["first_observation_index"]
        if idx > 1:
            d["autocall"]["first_observation_index"] = idx - 1
            d["autocall"]["levels_pct"] = (
                [d["autocall"]["levels_pct"][0]] + d["autocall"]["levels_pct"]
            )

    def autocall_level_up(d: dict) -> None:
        d["autocall"]["levels_pct"] = [lv + 5.0 for lv in d["autocall"]["levels_pct"]]

    def initial_level_up_1pct(d: dict) -> None:
        for u in d["underlyings"]:
            if u.get("initial_level"):
                u["initial_level"] *= 1.01

    def drop_last_observation(d: dict) -> None:
        if len(d["observation_dates"]) > 2:
            d["observation_dates"] = d["observation_dates"][:-1]
            d["payment_dates"] = d["payment_dates"][:-1]
            if len(d["autocall"]["levels_pct"]) > 1:
                d["autocall"]["levels_pct"] = d["autocall"]["levels_pct"][:-1]

    def monitoring_to_observation_dates(d: dict) -> None:
        d["knock_in"]["monitoring"] = (
            "observation_dates"
            if d["knock_in"]["monitoring"] == "at_maturity"
            else "at_maturity"
        )

    return [
        Perturbation("coupon.barrier_pct", "+5 percentage points", coupon_barrier_up),
        Perturbation("knock_in.barrier_pct", "-5 percentage points", knock_in_down),
        Perturbation(
            "coupon.rate_per_period_pct", "per-annum rate used as per-period",
            coupon_rate_per_annum,
        ),
        Perturbation("coupon.memory", "flipped", memory_flipped),
        Perturbation(
            "autocall.first_observation_index", "one period too early",
            autocall_starts_one_early,
        ),
        Perturbation("autocall.levels_pct", "+5 percentage points", autocall_level_up),
        Perturbation("underlyings.initial_level", "+1%", initial_level_up_1pct),
        Perturbation(
            "observation_dates", "final observation dropped", drop_last_observation
        ),
        Perturbation(
            "knock_in.monitoring", "European <-> observation-date monitoring",
            monitoring_to_observation_dates,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-filings", type=int, default=3)
    parser.add_argument("--k", type=int, default=3, help="self-consistency samples")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-paths", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args()

    candidates = [
        json.loads(line)
        for line in (RESULTS / "candidates.jsonl").read_text().splitlines()
    ]
    if not candidates:
        print("no candidates; run 01_build_candidate_pool.py first", file=sys.stderr)
        return 1

    # Prefer the smallest filings: the slice is about proving the chain, and the
    # free-tier quota is the binding constraint.
    candidates.sort(key=lambda r: r["approx_tokens_clean"])
    selected = candidates[: args.n_filings]

    provider = get_provider("gemini", model=args.model, temperature=args.temperature)
    print(f"model {args.model}  k={args.k}  temperature={args.temperature}")
    print(f"paths {args.n_paths:,}  seed {args.seed}\n")

    records: list[dict] = []
    sensitivity: dict[str, list[float]] = {}

    for rec in selected:
        label = f"{rec['issuer_stratum'][:28]} {rec['filing_date']}"
        print("=" * 78)
        print(f"{label}   {rec['approx_tokens_clean']:,} tokens")
        print(f"  {rec['url']}")

        text = (CLEANED / f"{rec['accession']}_{rec['document']}.txt").read_text()
        prompt = build_prompt(text)

        started = time.monotonic()
        calls = provider.sample(prompt, args.k)
        elapsed = time.monotonic() - started
        ok_calls = [c for c in calls if c.ok]
        in_tokens = sum(c.input_tokens or 0 for c in ok_calls)
        out_tokens = sum(c.output_tokens or 0 for c in ok_calls)
        think_tokens = sum(c.thinking_tokens or 0 for c in ok_calls)
        cached_tokens = sum(c.cached_input_tokens or 0 for c in ok_calls)
        print(
            f"  {len(ok_calls)}/{args.k} samples returned  "
            f"{in_tokens:,} in ({cached_tokens:,} cached) / {out_tokens:,} out "
            f"+ {think_tokens:,} reasoning tokens  {elapsed:.1f}s"
        )
        for call in calls:
            if call.error:
                print(f"    sample {call.seed_index} failed: {call.error[:110]}")
        if not ok_calls:
            records.append({"filing": rec, "status": "all samples failed"})
            continue

        payloads = [c.raw.model_dump(mode="json") for c in ok_calls]
        votes = vote_fields(payloads)
        weakest = votes.weakest_field()
        print(f"  field agreement (k={len(payloads)}):")
        for path, vote in votes.votes.items():
            flag = "  <- weakest" if weakest and path == weakest[0] else ""
            print(f"    {path:36s} {vote.agreement:.2f}  ({vote.distinct_values} distinct){flag}")

        chosen = majority_payload(payloads)
        terms, violations = to_note_terms(
            type(ok_calls[0].raw).model_validate(chosen)
        )
        if terms is None:
            print(f"  schema conversion FAILED ({len(violations)} violations):")
            for v in violations[:6]:
                print(f"    ! {v}")
            records.append(
                {
                    "filing": rec,
                    "status": "schema violation",
                    "violations": [str(v) for v in violations],
                    "confidence": votes.confidence(),
                }
            )
            continue
        print("  schema conversion OK")

        try:
            market = build_market_inputs(terms)
        except (UnknownUnderlyingError, MarketDataError) as exc:
            print(f"  market data unavailable: {exc}")
            records.append({"filing": rec, "status": f"market data: {exc}"})
            continue

        report = validate_terms(terms, initial_level_closes=initial_level_closes(terms))
        print(
            f"  validation: {len(report.hard)} hard, {len(report.soft)} soft"
            f"   confidence {votes.confidence(soft_warnings=len(report.soft)):.2f}"
        )
        for finding in report.findings[:6]:
            print(f"    {finding}")

        print(
            f"  market: {', '.join(market.tickers)}  "
            f"vol {[round(float(v), 3) for v in market.vols]}  "
            f"q {[round(float(q), 4) for q in market.div_yields]}"
        )

        try:
            base = price_pair(
                terms, terms, market, seed=args.seed, n_paths=args.n_paths
            )
        except UnsupportedProductError as exc:
            print(f"  not priceable: {exc}")
            records.append({"filing": rec, "status": f"unpriceable: {exc}"})
            continue

        print(
            f"  price {base.gold.price_per_1000:9.4f} per $1000 "
            f"(+/- {base.gold.std_error_per_1000:.4f} iid-equivalent)   "
            f"self-Delta {base.signed_deviation:+.1e}"
        )

        print("  per-field price sensitivity (controlled perturbation):")
        per_field: dict[str, float] = {}
        for pert in _perturbations():
            dump = copy.deepcopy(terms.model_dump(mode="json"))
            pert.apply(dump)
            try:
                perturbed = NoteTerms.model_validate(dump)
            except Exception as exc:  # noqa: BLE001 - perturbation may be invalid
                print(f"    {pert.name:36s} skipped (invalid: {type(exc).__name__})")
                continue
            if perturbed.model_dump() == terms.model_dump():
                print(f"    {pert.name:36s} no-op on this filing")
                continue
            try:
                pair = price_pair(
                    terms, perturbed, market, seed=args.seed, n_paths=args.n_paths
                )
            except UnsupportedProductError as exc:
                print(f"    {pert.name:36s} unpriceable ({exc})")
                continue
            per_field[pert.name] = pair.signed_deviation
            sensitivity.setdefault(pert.name, []).append(abs(pair.signed_deviation))
            print(
                f"    {pert.name:36s} {pert.description:34s} "
                f"Delta {pair.signed_deviation:+9.3f}  ({100 * pair.relative_deviation:+6.2f}%)"
            )

        records.append(
            {
                "filing": rec,
                "status": "ok",
                "model": args.model,
                "k": args.k,
                "n_ok_samples": len(ok_calls),
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "thinking_tokens": think_tokens,
                "cached_input_tokens": cached_tokens,
                "latency_s": round(elapsed, 2),
                "agreement": {p: v.agreement for p, v in votes.votes.items()},
                "confidence": votes.confidence(soft_warnings=len(report.soft)),
                "hard_findings": [str(f) for f in report.hard],
                "soft_findings": [str(f) for f in report.soft],
                "price_per_1000": base.gold.price_per_1000,
                "std_error_per_1000": base.gold.std_error_per_1000,
                "terms": terms.model_dump(mode="json"),
                "market": {
                    "tickers": list(market.tickers),
                    "spots": [float(s) for s in market.spots],
                    "vols": [float(v) for v in market.vols],
                    "div_yields": [float(q) for q in market.div_yields],
                    "meta": market.meta,
                },
                "field_deviation": per_field,
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "vertical_slice.json"
    out.write_text(json.dumps(records, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"wrote {out.relative_to(ROOT)}")
    priced = [r for r in records if r.get("status") == "ok"]
    print(f"priced {len(priced)}/{len(selected)} filings")

    if sensitivity:
        print("\nper-field |Delta| per $1000, across priced filings (ranking):")
        rows = sorted(
            ((k, statistics.median(v), max(v), len(v)) for k, v in sensitivity.items()),
            key=lambda r: r[1],
            reverse=True,
        )
        print(f"  {'field':38s} {'median':>9} {'max':>9} {'n':>3}")
        for name, med, mx, n in rows:
            print(f"  {name:38s} {med:9.3f} {mx:9.3f} {n:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
