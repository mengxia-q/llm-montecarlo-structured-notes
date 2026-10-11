"""Deterministic Monte Carlo valuation template.

Model assumptions (stated in the paper's Pricing Template section):

* Multi-dimensional geometric Brownian motion under the risk-neutral measure,
  with correlation from the realised daily log-return correlation matrix.
* Volatility and correlation estimated from the 252 trading days ending on the
  pricing date.
* Growth and discounting use forward rates implied by the FRED Treasury zero
  curve; dividend yields are constant per underlying.
* No issuer credit spread.  Notes are unsecured issuer debt and a spread would
  lower the level of every price, but it is held identical across the paired
  valuations and therefore cancels in the price-deviation measurement.

Variance reduction uses scrambled Sobol quasi-Monte Carlo rather than the
antithetic variates originally planned; see the discussion in ``src.pricing.rng``
for the measurement that motivated the change.

Because performance is only observed on the scheduled observation dates, the
simulation steps exactly from one observation date to the next; no finer grid is
needed and none is used.  Products with continuous/daily barrier monitoring fall
outside the template and are rejected rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import NoteTerms
from .market import MarketInputs
from .payoff import PayoffSpec, discounted_cashflows
from .rng import CommonRandomNumbers

#: Day-count used to turn calendar dates into year fractions.  ACT/365 fixed.
DAYS_PER_YEAR = 365.0


class UnsupportedProductError(ValueError):
    """Raised when the terms fall outside the template's coverage."""


@dataclass(frozen=True)
class PricingResult:
    price_per_1000: float
    """Fair value per $1000 of notional - the unit used for every reported
    price deviation, so that filings with $10, $100 and $1000 units compare."""

    std_error_per_1000: float
    """Standard error of the simulated mean, same units.

    For ``generator="iid"`` this is the usual sample standard error (computed on
    antithetic pair averages when antithetic sampling is on, since the paired
    draws are dependent).

    For ``generator="sobol"`` the points are not independent, so the sample
    standard error is only an *iid-equivalent upper bound*: scrambled Sobol
    converges faster than the bound suggests.  The honest error estimate for
    quasi-Monte Carlo comes from replicating independent scrambles, which is what
    ``experiments/08_convergence.py`` does across seeds.
    """

    price_per_unit: float
    """Fair value in the filing's own notional unit."""

    n_paths: int
    generator: str
    antithetic: bool
    seed: int

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"{self.price_per_1000:.4f} +/- {self.std_error_per_1000:.4f} per $1000 "
            f"({self.n_paths:,} {self.generator} paths, seed={self.seed})"
        )


def year_fractions(terms: NoteTerms, dates) -> np.ndarray:
    return np.array(
        [(d - terms.pricing_date).days / DAYS_PER_YEAR for d in dates], dtype=float
    )


def build_payoff_spec(terms: NoteTerms) -> PayoffSpec:
    """Translate schema terms into decimal-unit payoff parameters."""
    if terms.knock_in.monitoring == "continuous_daily":
        raise UnsupportedProductError(
            "continuous/daily knock-in monitoring is outside the template's coverage"
        )

    n_obs = terms.n_observations
    levels = np.full(n_obs, np.nan)
    for i in range(n_obs):
        level = terms.autocall_level_at(i)
        if level is not None:
            levels[i] = level / 100.0

    return PayoffSpec(
        coupon_rate=terms.coupon.rate_per_period_pct / 100.0,
        coupon_barrier=terms.coupon.barrier_pct / 100.0,
        memory=terms.coupon.memory,
        autocall_levels=levels,
        knock_in_barrier=terms.knock_in.barrier_pct / 100.0,
        knock_in_at_maturity=terms.knock_in.monitoring == "at_maturity",
        downside_participation=terms.downside_participation_pct / 100.0,
    )


def simulate_performance(
    terms: NoteTerms,
    market: MarketInputs,
    crn: CommonRandomNumbers,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Worst-of performance ratios at each observation date.

    Paths start at the market closing level on the pricing date and performance
    is divided by the *extracted* initial level, which is what gives the
    ``initial_level`` field price sensitivity.

    ``z`` overrides the normals taken from ``crn`` (one chunk of
    ``crn.iter_draws``); by default the full ``crn.draw`` block is used.
    """
    obs_times = year_fractions(terms, terms.observation_dates)
    if np.any(np.diff(obs_times) <= 0):
        raise UnsupportedProductError("observation dates must be strictly increasing")
    if obs_times[0] <= 0:
        raise UnsupportedProductError(
            "the first observation date must fall after the pricing date"
        )

    n_assets = len(terms.underlyings)
    n_obs = len(obs_times)

    initial_levels = np.array(
        [
            u.initial_level if u.initial_level is not None else market.spots[j]
            for j, u in enumerate(terms.underlyings)
        ],
        dtype=float,
    )
    if np.any(initial_levels <= 0):
        raise UnsupportedProductError("initial levels must be positive")

    vols = np.asarray(market.vols, dtype=float)[:n_assets]
    divs = np.asarray(market.div_yields, dtype=float)[:n_assets]
    spots = np.asarray(market.spots, dtype=float)[:n_assets]

    fwd = market.curve.forward_rates(obs_times)
    prev = np.concatenate([[0.0], obs_times[:-1]])
    dt = obs_times - prev

    if z is None:
        z = crn.draw(n_obs, n_assets)
    chol = market.cholesky()[:n_assets, :n_assets]
    correlated = z @ chol.T  # (n_paths, n_obs, n_assets)

    drift = (fwd[:, None] - divs[None, :] - 0.5 * vols[None, :] ** 2) * dt[:, None]
    diffusion = vols[None, None, :] * np.sqrt(dt)[None, :, None] * correlated
    log_paths = np.cumsum(drift[None, :, :] + diffusion, axis=1)

    levels = spots[None, None, :] * np.exp(log_paths)
    ratios = levels / initial_levels[None, None, :]
    return ratios.min(axis=2) if n_assets > 1 else ratios[:, :, 0]


def _path_samples(pv: np.ndarray, crn: CommonRandomNumbers) -> np.ndarray:
    """Independent samples behind the estimator.

    Antithetic pairs are dependent; the standard error must be computed on the
    pair averages, otherwise it is understated.  A block (or chunk) from an
    antithetic generator is laid out as ``[h, -h]``.
    """
    if crn.antithetic:
        half = len(pv) // 2
        return 0.5 * (pv[:half] + pv[half:])
    return pv


def price_note(
    terms: NoteTerms,
    market: MarketInputs,
    crn: CommonRandomNumbers,
    *,
    chunk_paths: int | None = None,
) -> PricingResult:
    """Fair value of one note, normalised to a per-$1000 basis.

    ``chunk_paths`` simulates the paths in consecutive blocks of that many,
    accumulating the payoff sum and sum of squares, so that path counts whose
    full normal block would not fit in memory (the convergence study runs to
    2^20) can be priced.  The point set is the same as unchunked - one Sobol
    sequence, one pseudo-random stream - so the price agrees with the unchunked
    value up to floating-point summation order.  ``None`` (the default) keeps
    the single-block path, which is what paired valuations use.
    """
    if len(market.tickers) != len(terms.underlyings):
        raise ValueError(
            f"market inputs cover {len(market.tickers)} underlyings but the terms "
            f"declare {len(terms.underlyings)}"
        )

    spec = build_payoff_spec(terms)
    pay_times = year_fractions(terms, terms.payment_dates)
    if np.any(pay_times <= 0):
        raise UnsupportedProductError("payment dates must fall after the pricing date")
    dfs = np.asarray(market.curve.discount(pay_times), dtype=float)

    if chunk_paths is None:
        performance = simulate_performance(terms, market, crn)
        samples = _path_samples(discounted_cashflows(performance, spec, dfs), crn)
        mean = float(samples.mean())
        stderr = (
            float(samples.std(ddof=1) / np.sqrt(len(samples))) if len(samples) > 1 else 0.0
        )
    else:
        # Shifted sums around the first chunk's mean keep the variance
        # numerically stable at large path counts.
        n, total, total_sq, shift = 0, 0.0, 0.0, None
        n_obs, n_assets = terms.n_observations, len(terms.underlyings)
        for z in crn.iter_draws(n_obs, n_assets, chunk_paths):
            performance = simulate_performance(terms, market, crn, z=z)
            samples = _path_samples(discounted_cashflows(performance, spec, dfs), crn)
            if shift is None:
                shift = float(samples.mean())
            d = samples - shift
            n += len(d)
            total += float(d.sum())
            total_sq += float((d * d).sum())
        mean = shift + total / n
        var = (total_sq - total * total / n) / (n - 1) if n > 1 else 0.0
        stderr = float(np.sqrt(max(var, 0.0) / n)) if n > 1 else 0.0

    return PricingResult(
        price_per_1000=mean * 1000.0,
        std_error_per_1000=stderr * 1000.0,
        price_per_unit=mean * terms.notional_per_unit,
        n_paths=crn.n_paths,
        generator=crn.generator,
        antithetic=crn.antithetic,
        seed=crn.seed,
    )
