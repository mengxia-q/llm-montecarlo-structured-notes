"""End-to-end price deviation: the paper's primary quality metric.

    Delta = V(theta_hat) - V(theta*)

in dollars per $1000 of notional, where both valuations use the same market data
and the same random numbers.  With identical parameters Delta is exactly zero, so
everything measured is attributable to extraction error rather than to simulation
noise.

Both the signed and the absolute deviation are reported.  The signed figure
answers a question the absolute one hides: whether extraction errors are
directionally biased - systematically flattering or penalising the note - which
matters to anyone using the output for independent price verification.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pricing import CommonRandomNumbers, MarketInputs, PricingResult, price_note
from ..pricing.rng import Generator
from ..schema import NoteTerms

#: Production path count.  A power of two, because Sobol point sets are balanced
#: only at powers of two; 2^17 = 131,072 is the nearest to the planned 100,000.
DEFAULT_N_PATHS = 131_072


@dataclass(frozen=True)
class PairedValuation:
    gold: PricingResult
    extracted: PricingResult

    @property
    def signed_deviation(self) -> float:
        """``V(extracted) - V(gold)`` per $1000 of notional."""
        return self.extracted.price_per_1000 - self.gold.price_per_1000

    @property
    def abs_deviation(self) -> float:
        return abs(self.signed_deviation)

    @property
    def relative_deviation(self) -> float:
        """Signed deviation as a fraction of the gold-standard price."""
        if self.gold.price_per_1000 == 0:
            return float("nan")
        return self.signed_deviation / self.gold.price_per_1000

    def __str__(self) -> str:  # pragma: no cover - reporting only
        return (
            f"gold {self.gold.price_per_1000:9.4f}  extracted "
            f"{self.extracted.price_per_1000:9.4f}  delta "
            f"{self.signed_deviation:+9.4f} per $1000 "
            f"({100 * self.relative_deviation:+.3f}%)"
        )


def _crn_for(
    gold: NoteTerms,
    extracted: NoteTerms,
    *,
    seed: int,
    n_paths: int,
    generator: Generator,
) -> CommonRandomNumbers:
    """A single random-number configuration that covers both parameter sets.

    The draw shape must be identical for the two valuations, or they would not
    share random numbers at all; it is therefore sized to the larger of the two on
    each axis - and to exactly that, not to a generous default.  For Sobol the
    shape sets the point-set dimension, and padding a 12-dimensional problem out
    to 256 dimensions wastes the well-distributed leading dimensions and makes
    each draw more than an order of magnitude more expensive.
    """
    max_steps = max(gold.n_observations, extracted.n_observations)
    max_assets = max(len(gold.underlyings), len(extracted.underlyings))
    return CommonRandomNumbers(
        seed=seed,
        n_paths=n_paths,
        generator=generator,
        max_steps=max_steps,
        max_assets=max_assets,
    )


def price_pair(
    gold: NoteTerms,
    extracted: NoteTerms,
    market: MarketInputs,
    *,
    seed: int = 20260101,
    n_paths: int = DEFAULT_N_PATHS,
    generator: Generator = "sobol",
    extracted_market: MarketInputs | None = None,
) -> PairedValuation:
    """Value one filing twice and return the deviation.

    ``market`` is built from the gold-standard underlyings and is used for both
    valuations, so that market-data choices cancel.  ``extracted_market`` is only
    needed when the extraction got the basket itself wrong and a second market
    state is unavoidable; the deviation then mixes a market-data change with the
    parameter change, which the caller should record.
    """
    crn = _crn_for(gold, extracted, seed=seed, n_paths=n_paths, generator=generator)
    return PairedValuation(
        gold=price_note(gold, market, crn),
        extracted=price_note(extracted, extracted_market or market, crn),
    )
