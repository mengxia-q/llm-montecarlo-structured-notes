"""A relaxed mirror of ``src.schema`` for model output, plus strict conversion.

``NoteTerms`` enforces structural invariants (payment and observation dates of
equal length, one autocall level per callable date, ...).  Those invariants are
exactly what a model gets wrong, so making them hard errors at parse time would
crash the pipeline on the most interesting cases instead of measuring them.

The model is therefore asked for ``RawNoteTerms``, which has the same field names
and no cross-field validation, and ``to_note_terms`` reports every invariant
violation it finds.  The evaluation reports the share of filings that fail this
conversion as a separate figure, rather than dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schema import NoteTerms


class RawUnderlying(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(
        default=None, description="Underlying name exactly as written in the filing."
    )
    ticker: str | None = Field(
        default=None, description="Ticker if the filing states one, else null."
    )
    initial_level: float | None = Field(
        default=None,
        description="Initial level / Starting Value / Initial Underlier Level, to the "
        "precision printed in the filing.",
    )


class RawAutocall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    first_observation_index: int | None = Field(
        default=None,
        description="1-based position in observation_dates of the FIRST date on which "
        "the note can be automatically called. If every observation date is callable "
        "this is 1.",
    )
    levels_pct: list[float] = Field(
        default_factory=list,
        description="Autocall trigger level for each callable observation date, in "
        "percent of the initial level, chronological. Exactly one entry per callable "
        "date, so a step-down schedule is written out in full.",
    )


class RawCoupon(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rate_per_period_pct: float | None = Field(
        default=None,
        description="Contingent coupon actually paid per period, in percent of "
        "notional. If the filing quotes a per-annum rate, divide by the number of "
        "periods per year.",
    )
    barrier_pct: float | None = Field(
        default=None, description="Coupon barrier, in percent of the initial level."
    )
    memory: bool | None = Field(
        default=None,
        description="True only if unpaid coupons accumulate and are paid later when "
        "the barrier is met again (memory / snowball). False otherwise.",
    )


class RawKnockIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    barrier_pct: float | None = Field(
        default=None,
        description="Downside protection barrier in percent of the initial level "
        "(Threshold Value / Trigger Level / Knock-In Level / Downside Threshold).",
    )
    monitoring: Literal["at_maturity", "observation_dates", "continuous_daily"] | None = Field(
        default=None,
        description="'at_maturity' if the barrier is tested only on the final "
        "valuation date; 'observation_dates' if tested on each scheduled observation "
        "date; 'continuous_daily' if tested on every trading day.",
    )


class RawNoteTerms(BaseModel):
    """What the model is asked to return.  Every field may be null."""

    model_config = ConfigDict(extra="ignore")

    issuer: str | None = None
    notional_per_unit: float | None = Field(
        default=None,
        description="Principal amount per unit as filed: 10, 100 or 1000, not a "
        "normalised value.",
    )
    currency: str | None = Field(default=None, description="ISO code, e.g. USD.")

    pricing_date: date | None = Field(default=None, description="YYYY-MM-DD.")
    strike_date: date | None = Field(
        default=None,
        description="YYYY-MM-DD. The date the initial levels were set, ONLY if the "
        "filing names a strike date different from the pricing/trade date; else null.",
    )
    issue_date: date | None = Field(default=None, description="YYYY-MM-DD.")
    maturity_date: date | None = Field(default=None, description="YYYY-MM-DD.")

    underlyings: list[RawUnderlying] = Field(default_factory=list)
    performance_type: Literal["single", "worst_of"] | None = Field(
        default=None,
        description="'single' for one underlying, 'worst_of' if the payoff depends on "
        "the worst performer of a basket.",
    )

    observation_dates: list[date] = Field(
        default_factory=list,
        description="Every coupon observation / valuation date, chronological, "
        "YYYY-MM-DD. The last entry is the final valuation date.",
    )
    payment_dates: list[date] = Field(
        default_factory=list,
        description="The payment date matching each observation date, same length and "
        "order as observation_dates.",
    )

    autocall: RawAutocall = Field(default_factory=RawAutocall)
    coupon: RawCoupon = Field(default_factory=RawCoupon)
    knock_in: RawKnockIn = Field(default_factory=RawKnockIn)

    downside_participation_pct: float | None = Field(
        default=None,
        description="Downside gearing below the barrier in percent; 100 for the usual "
        "1-to-1 exposure.",
    )
    estimated_value_per_unit: float | None = Field(
        default=None,
        description="The issuer's own estimated value per unit on the pricing date. "
        "Null if the filing gives only a range.",
    )


@dataclass(frozen=True)
class SchemaViolation:
    """One reason a raw extraction could not become a priceable ``NoteTerms``."""

    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - reporting only
        return f"{self.field}: {self.message}"


def to_note_terms(raw: RawNoteTerms) -> tuple[NoteTerms | None, list[SchemaViolation]]:
    """Promote a raw extraction to strict ``NoteTerms``.

    Returns ``(terms, [])`` on success, or ``(None, violations)`` listing every
    missing field and broken invariant.  Nothing is repaired or guessed: a
    silently patched parameter set would understate the price impact of the
    extraction error, which is the quantity the paper measures.
    """
    violations: list[SchemaViolation] = []

    required = {
        "issuer": raw.issuer,
        "notional_per_unit": raw.notional_per_unit,
        "pricing_date": raw.pricing_date,
        "maturity_date": raw.maturity_date,
        "performance_type": raw.performance_type,
        "coupon.rate_per_period_pct": raw.coupon.rate_per_period_pct,
        "coupon.barrier_pct": raw.coupon.barrier_pct,
        "coupon.memory": raw.coupon.memory,
        "knock_in.barrier_pct": raw.knock_in.barrier_pct,
        "knock_in.monitoring": raw.knock_in.monitoring,
        "autocall.first_observation_index": raw.autocall.first_observation_index,
    }
    for name, value in required.items():
        if value is None:
            violations.append(SchemaViolation(name, "missing"))

    if not raw.underlyings:
        violations.append(SchemaViolation("underlyings", "empty"))
    if not raw.observation_dates:
        violations.append(SchemaViolation("observation_dates", "empty"))
    if not raw.payment_dates:
        violations.append(SchemaViolation("payment_dates", "empty"))
    if not raw.autocall.levels_pct:
        violations.append(SchemaViolation("autocall.levels_pct", "empty"))

    for i, underlying in enumerate(raw.underlyings):
        if underlying.name is None:
            violations.append(SchemaViolation(f"underlyings[{i}].name", "missing"))
        if underlying.initial_level is None:
            violations.append(
                SchemaViolation(f"underlyings[{i}].initial_level", "missing")
            )

    if violations:
        return None, violations

    payload = raw.model_dump()
    payload["currency"] = (raw.currency or "USD").upper()[:3]
    if raw.downside_participation_pct is None:
        payload["downside_participation_pct"] = 100.0

    try:
        return NoteTerms.model_validate(payload), []
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "<model>"
            violations.append(SchemaViolation(location, error["msg"]))
        return None, violations
