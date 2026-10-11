"""Pricing-driving parameter schema for autocallable / Phoenix structured notes.

Field naming follows FpML / ISDA CDM conventions where a direct analogue exists,
but the schema deliberately covers only the minimum set required to price the
product with the deterministic template in ``src/pricing``.

Design notes
------------
* ``notional_per_unit`` is kept at its as-filed value (issuers use $10, $100 and
  $1000 units).  All valuation output is normalised to a per-$1000 basis by the
  pricing layer so that price deviations are comparable across filings.
* Percentages are always expressed relative to the underlying's initial level
  ("Starting Value"), in percent units: 70 means 70%, not 0.70.
* ``autocall.levels_pct`` is a sequence with one entry per callable observation
  date, which natively supports step-down call schedules.
* ``estimated_value_per_unit`` is annotated but not used in the conference-paper
  analysis; it is retained for future work.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Pct = Annotated[float, Field(ge=0.0, le=1000.0)]

PerformanceType = Literal["single", "worst_of"]
KnockInMonitoring = Literal["at_maturity", "observation_dates", "continuous_daily"]

#: Fields that participate in the document-level confidence score (paper: Voting and
#: Confidence) and in the per-field price-sensitivity analysis (Which Errors Matter).
CRITICAL_FIELDS: tuple[str, ...] = (
    "underlyings.initial_level",
    "observation_dates",
    "autocall.levels_pct",
    "autocall.first_observation_index",
    "coupon.rate_per_period_pct",
    "coupon.barrier_pct",
    "coupon.memory",
    "knock_in.barrier_pct",
    "maturity_date",
)


class Underlying(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Underlying name exactly as written in the filing.")
    ticker: str | None = Field(
        default=None,
        description="Bloomberg-style or exchange ticker if the filing states one, else null.",
    )
    initial_level: float | None = Field(
        default=None,
        gt=0,
        description='Initial level / "Starting Value" / "Initial Underlier Level", '
        "to the precision printed in the filing.",
    )


class Autocall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_observation_index: int = Field(
        ge=1,
        description="1-based index into observation_dates of the first date on which "
        "the note may be automatically called.",
    )
    levels_pct: list[Pct] = Field(
        min_length=1,
        description="Autocall trigger level per callable observation date, in percent "
        "of the initial level. One entry per callable date, in chronological order. "
        "The callable window starts at first_observation_index and runs for as many "
        "dates as there are entries here, so a note that is callable on its interim "
        "valuation dates but not on the final one has one fewer entry than it has "
        "observation dates from first_observation_index onwards.",
    )


class Coupon(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_per_period_pct: float = Field(
        ge=0,
        description="Contingent coupon paid per period, in percent of notional "
        "(NOT the per-annum rate).",
    )
    barrier_pct: Pct = Field(
        description="Coupon barrier in percent of the initial level."
    )
    memory: bool = Field(
        description="True if unpaid coupons accrue and are paid on a later date on "
        'which the barrier is met ("memory" / "snowball" coupon).',
    )


class KnockIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    barrier_pct: Pct = Field(
        description="Downside protection barrier in percent of the initial level "
        '("Threshold Value", "Trigger Level", "Knock-In Level").',
    )
    monitoring: KnockInMonitoring = Field(
        description="When the barrier is tested: only at maturity (European), on the "
        "scheduled observation dates, or continuously/daily.",
    )


class NoteTerms(BaseModel):
    """The complete set of parameters needed to price one note."""

    model_config = ConfigDict(extra="forbid")

    issuer: str
    notional_per_unit: float = Field(gt=0, description="Principal amount per unit, as filed.")
    currency: str = Field(default="USD", min_length=3, max_length=3)

    pricing_date: date = Field(description="Trade/pricing date on which terms were set.")
    strike_date: date | None = Field(
        default=None,
        description="Date on which the initial levels were fixed, when the filing "
        "names one distinct from the pricing date (typically the business day "
        "before). Null when the initial levels are the pricing-date closes.",
    )
    issue_date: date | None = Field(default=None, description="Settlement/issue date.")
    maturity_date: date

    underlyings: list[Underlying] = Field(min_length=1)
    performance_type: PerformanceType = Field(
        description="'single' for one underlying; 'worst_of' if payoff depends on the "
        "worst performer of a basket.",
    )

    observation_dates: list[date] = Field(
        min_length=1,
        description="Coupon observation / valuation dates in chronological order. The "
        "final entry is the final valuation date.",
    )
    payment_dates: list[date] = Field(
        min_length=1,
        description="Payment date corresponding to each observation date, same length "
        "and order as observation_dates.",
    )

    autocall: Autocall
    coupon: Coupon
    knock_in: KnockIn

    downside_participation_pct: float = Field(
        default=100.0,
        ge=0,
        description="Downside gearing below the barrier, in percent. 100 means 1-to-1 "
        "exposure, which is the overwhelmingly common case.",
    )

    estimated_value_per_unit: float | None = Field(
        default=None,
        description="Issuer's own estimated value per unit on the pricing date. "
        "Annotated for completeness; not used in the pricing-deviation analysis. "
        "Null if the filing gives only a range (preliminary term sheets).",
    )

    # ---- structural invariants (always enforced; distinct from the business
    # ---- rule layer in src/validate, which is allowed to be violated) ----
    @model_validator(mode="after")
    def _check_structure(self) -> "NoteTerms":
        if len(self.payment_dates) != len(self.observation_dates):
            raise ValueError(
                f"payment_dates ({len(self.payment_dates)}) and observation_dates "
                f"({len(self.observation_dates)}) must have equal length"
            )
        n_obs = len(self.observation_dates)
        if self.autocall.first_observation_index > n_obs:
            raise ValueError(
                f"autocall.first_observation_index ({self.autocall.first_observation_index}) "
                f"exceeds the number of observation dates ({n_obs})"
            )
        # The callable window is defined by the number of levels, not by the end of
        # the schedule.  Filings routinely make the note callable on the *interim*
        # valuation dates only, leaving the final valuation date to the maturity
        # barrier rule, in which case there is one level fewer than there are
        # observation dates from first_observation_index onwards.
        window_end = self.autocall.first_observation_index - 1 + len(
            self.autocall.levels_pct
        )
        if window_end > n_obs:
            raise ValueError(
                f"autocall.levels_pct has {len(self.autocall.levels_pct)} entries "
                f"starting at observation {self.autocall.first_observation_index}, "
                f"which runs past the {n_obs} observation dates"
            )
        if self.performance_type == "single" and len(self.underlyings) != 1:
            raise ValueError("performance_type 'single' requires exactly one underlying")
        if self.performance_type == "worst_of" and len(self.underlyings) < 2:
            raise ValueError("performance_type 'worst_of' requires at least two underlyings")
        return self

    # ---- convenience accessors used by the pricing and eval layers ----
    @property
    def n_observations(self) -> int:
        return len(self.observation_dates)

    @property
    def final_valuation_date(self) -> date:
        return self.observation_dates[-1]

    @property
    def initial_level_date(self) -> date:
        """The date whose closes the filed initial levels are.

        Valuation still starts from the pricing-date close (paper: Pricing Template); a strike
        date only changes which close the initial-level cross-check compares to.
        """
        return self.strike_date or self.pricing_date

    def callable_mask(self) -> list[bool]:
        """One flag per observation date: is the note callable on that date?"""
        start = self.autocall.first_observation_index - 1
        end = start + len(self.autocall.levels_pct)
        return [start <= i < end for i in range(self.n_observations)]

    def autocall_level_at(self, i: int) -> float | None:
        """Autocall trigger (percent) at 0-based observation index ``i``, or None.

        Returns ``None`` both before the callable window opens and after it
        closes, the latter being the common case where the final valuation date is
        not a call date.
        """
        start = self.autocall.first_observation_index - 1
        offset = i - start
        if not 0 <= offset < len(self.autocall.levels_pct):
            return None
        return self.autocall.levels_pct[offset]
