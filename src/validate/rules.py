"""Deterministic business-rule validation of an extracted parameter set.

Two severities (paper: Business Rules), so that unusual-but-legitimate terms are not
discarded along with genuine extraction errors:

**hard** - the parameter set is internally impossible or unpriceable.  Triggers
one validator-guided re-extraction, with the findings fed back in natural
language (see ``src.extract.prompt.build_prompt``).

**soft** - the parameter set is legal but unusual.  Nothing is re-extracted; the
document's confidence score is multiplied by a penalty, which pushes borderline
filings towards manual review instead of silently through.

The rules are deliberately mechanical.  Anything requiring judgement belongs in
manual review, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np

from ..market.tickers import UnknownUnderlyingError, resolve_underlying
from ..schema import NoteTerms


class Severity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.rule}: {self.message}"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def hard(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.HARD]

    @property
    def soft(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.SOFT]

    @property
    def ok(self) -> bool:
        return not self.hard

    def messages(self) -> list[str]:
        """Natural-language findings for the re-extraction prompt."""
        return [str(f) for f in self.hard]


#: Barrier levels outside this range are not plausible for this product class.
BARRIER_RANGE_PCT = (1.0, 150.0)

#: Tolerance for the initial-level cross-check against the strike-date close.
INITIAL_LEVEL_TOLERANCE = 0.005

#: Scheduled observation intervals are expected to be regular.  A gap more than
#: this multiple of the median gap is flagged as a likely missed or duplicated row.
INTERVAL_IRREGULARITY_FACTOR = 1.6


def _check_dates(terms: NoteTerms, out: list[Finding]) -> None:
    obs = terms.observation_dates
    if any(b <= a for a, b in zip(obs, obs[1:])):
        out.append(
            Finding(
                "observation_dates_increasing",
                Severity.HARD,
                "observation dates are not strictly increasing, which usually means "
                "a schedule table was read out of order or a row was duplicated",
            )
        )

    outside = [d for d in obs if not (terms.pricing_date < d <= terms.maturity_date)]
    if outside:
        out.append(
            Finding(
                "observation_dates_within_term",
                Severity.HARD,
                f"{len(outside)} observation date(s) fall outside "
                f"({terms.pricing_date}, {terms.maturity_date}], first {outside[0]}",
            )
        )

    early = [
        (o, p) for o, p in zip(obs, terms.payment_dates) if p < o
    ]
    if early:
        out.append(
            Finding(
                "payment_after_observation",
                Severity.HARD,
                f"{len(early)} payment date(s) precede their observation date, "
                f"first {early[0][1]} < {early[0][0]}",
            )
        )

    if terms.maturity_date <= terms.pricing_date:
        out.append(
            Finding(
                "maturity_after_pricing",
                Severity.HARD,
                f"maturity {terms.maturity_date} is not after pricing "
                f"{terms.pricing_date}",
            )
        )

    if terms.strike_date is not None and terms.strike_date > terms.pricing_date:
        out.append(
            Finding(
                "strike_not_after_pricing",
                Severity.HARD,
                f"strike date {terms.strike_date} is after the pricing date "
                f"{terms.pricing_date}; initial levels are fixed on or before the "
                "date the terms are set",
            )
        )

    if len(obs) >= 3:
        gaps = np.diff([d.toordinal() for d in obs])
        median = float(np.median(gaps))
        if median > 0:
            odd = [
                (i, int(g))
                for i, g in enumerate(gaps)
                if g > INTERVAL_IRREGULARITY_FACTOR * median
                or g < median / INTERVAL_IRREGULARITY_FACTOR
            ]
            if odd:
                out.append(
                    Finding(
                        "observation_interval_regularity",
                        Severity.SOFT,
                        f"{len(odd)} observation interval(s) deviate from the median "
                        f"{median:.0f}-day spacing (first: {odd[0][1]} days after "
                        f"observation {odd[0][0] + 1}), suggesting a skipped or "
                        "duplicated schedule row",
                    )
                )


def _check_barriers(terms: NoteTerms, out: list[Finding]) -> None:
    low, high = BARRIER_RANGE_PCT
    named = {
        "coupon.barrier_pct": terms.coupon.barrier_pct,
        "knock_in.barrier_pct": terms.knock_in.barrier_pct,
    }
    for i, level in enumerate(terms.autocall.levels_pct):
        named[f"autocall.levels_pct[{i}]"] = level

    for name, value in named.items():
        if not low <= value <= high:
            out.append(
                Finding(
                    "barrier_plausible_range",
                    Severity.HARD,
                    f"{name} = {value} is outside the plausible range "
                    f"[{low}, {high}] percent of the initial level",
                )
            )

    levels = terms.autocall.levels_pct
    rises = [i for i, (a, b) in enumerate(zip(levels, levels[1:])) if b > a]
    if rises:
        out.append(
            Finding(
                "autocall_ladder_non_increasing",
                Severity.SOFT,
                f"autocall level rises at position {rises[0] + 1} "
                f"({levels[rises[0]]} -> {levels[rises[0] + 1]}); step-down schedules "
                "should be non-increasing",
            )
        )

    if terms.knock_in.barrier_pct > terms.coupon.barrier_pct:
        out.append(
            Finding(
                "knock_in_below_coupon_barrier",
                Severity.SOFT,
                f"knock-in barrier ({terms.knock_in.barrier_pct}%) is above the coupon "
                f"barrier ({terms.coupon.barrier_pct}%), which is unusual: principal "
                "protection normally extends further down than the coupon condition",
            )
        )

    if terms.coupon.rate_per_period_pct > 15.0:
        out.append(
            Finding(
                "coupon_rate_per_period",
                Severity.SOFT,
                f"coupon of {terms.coupon.rate_per_period_pct}% per period is very "
                "large; a per-annum rate may have been copied without dividing by "
                "the number of periods per year",
            )
        )


def _check_underlyings(terms: NoteTerms, out: list[Finding]) -> None:
    for i, underlying in enumerate(terms.underlyings):
        try:
            resolve_underlying(underlying.name, underlying.ticker)
        except UnknownUnderlyingError as exc:
            out.append(
                Finding(
                    "underlying_resolvable",
                    Severity.HARD,
                    f"underlyings[{i}] {exc}",
                )
            )

    n = len(terms.underlyings)
    if n > 1 and terms.performance_type != "worst_of":
        out.append(
            Finding(
                "performance_type_consistent",
                Severity.HARD,
                f"{n} underlyings were extracted but performance_type is "
                f"{terms.performance_type!r}",
            )
        )


def _check_initial_levels(
    terms: NoteTerms, spots: dict[str, float] | None, out: list[Finding]
) -> None:
    """Cross-check each initial level against the close on its fixing date.

    Cheap and unusually informative: the Starting Value is by construction the
    closing level on the strike date, or on the pricing date when the filing
    names no separate strike date, so any material disagreement is an extraction
    error in a field that directly scales every performance ratio.  Comparing
    against the pricing-date close regardless flagged correct extractions on
    strike-dated notes (1.6% on one TD filing's Nasdaq-100 level).
    """
    if not spots:
        return
    for i, underlying in enumerate(terms.underlyings):
        if underlying.initial_level is None:
            continue
        try:
            ticker = resolve_underlying(underlying.name, underlying.ticker).yahoo_ticker
        except UnknownUnderlyingError:
            continue
        close = spots.get(ticker)
        if close is None or close <= 0:
            continue
        relative = abs(underlying.initial_level / close - 1.0)
        if relative > INITIAL_LEVEL_TOLERANCE:
            out.append(
                Finding(
                    "initial_level_matches_close",
                    Severity.SOFT,
                    f"underlyings[{i}] initial level {underlying.initial_level:,.2f} "
                    f"differs from the {ticker} close of {close:,.2f} on "
                    f"{terms.initial_level_date} by {100 * relative:.2f}% "
                    f"(tolerance {100 * INITIAL_LEVEL_TOLERANCE:.1f}%)",
                )
            )


def validate_terms(
    terms: NoteTerms, *, initial_level_closes: dict[str, float] | None = None
) -> ValidationReport:
    """Run every rule over one parameter set.

    ``initial_level_closes`` maps market-data ticker to the closing level on
    ``terms.initial_level_date`` (the strike date if there is one, else the
    pricing date); pass it to enable the initial-level cross-check.
    """
    findings: list[Finding] = []
    _check_dates(terms, findings)
    _check_barriers(terms, findings)
    _check_underlyings(terms, findings)
    _check_initial_levels(terms, initial_level_closes, findings)
    return ValidationReport(findings)
