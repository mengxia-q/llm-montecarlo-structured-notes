"""Payoff rules for Phoenix autocallable notes with memory coupons and step-down
autocall levels.

The function is written against a pre-computed performance matrix so that it can
be unit-tested on hand-constructed deterministic paths (``tests/test_payoff.py``) without
touching the GBM simulator.

Payoff convention, per observation index ``i = 0 .. N-1``, with performance
``P_i = min_j S_j(t_i) / L_j`` (``L_j`` = initial level of underlying ``j``):

1. Contingent coupon: if ``P_i >= coupon_barrier`` the period coupon is paid on
   payment date ``i``; with a memory feature, all previously missed coupons are
   paid at the same time.
2. Automatic call: if date ``i`` is callable and ``P_i >= autocall_level_i``, the
   notional is repaid on payment date ``i`` and the note terminates.  Step 1 is
   evaluated first, so the coupon due on the call date is paid, which matches the
   wording in the filings ("you will receive the principal amount plus the
   Contingent Coupon Payment otherwise due").
3. Maturity: if never called, par is repaid when the barrier is respected,
   otherwise the investor is exposed to the decline with the stated
   participation.  Under observation-date monitoring a breach forfeits par only
   if the final performance is also below 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PayoffSpec:
    """Payoff parameters in decimal (not percent) units, per unit of notional 1.0."""

    coupon_rate: float
    """Coupon per period as a fraction of notional (0.025 = 2.5% of notional)."""

    coupon_barrier: float
    """Coupon barrier as a fraction of the initial level (0.70 = 70%)."""

    memory: bool

    autocall_levels: np.ndarray
    """One level (fraction of initial) per observation date; ``nan`` where the
    note is not callable on that date."""

    knock_in_barrier: float
    """Downside barrier as a fraction of the initial level."""

    knock_in_at_maturity: bool
    """True for European observation (only the final level matters); False if the
    barrier is tested on every scheduled observation date."""

    downside_participation: float = 1.0
    """1.0 = 1-to-1 downside exposure below the barrier."""


def discounted_cashflows(
    performance: np.ndarray,
    spec: PayoffSpec,
    discount_factors: np.ndarray,
) -> np.ndarray:
    """Present value per path, in units of notional = 1.0.

    Parameters
    ----------
    performance:
        ``(n_paths, n_obs)`` matrix of worst-of performance ratios.
    spec:
        Payoff parameters.
    discount_factors:
        ``(n_obs,)`` discount factors to each *payment* date.

    Returns
    -------
    ``(n_paths,)`` array of path present values.
    """
    performance = np.asarray(performance, dtype=float)
    if performance.ndim != 2:
        raise ValueError(f"performance must be 2-D, got shape {performance.shape}")
    n_paths, n_obs = performance.shape
    if spec.autocall_levels.shape != (n_obs,):
        raise ValueError(
            f"autocall_levels must have shape ({n_obs},), got {spec.autocall_levels.shape}"
        )
    if discount_factors.shape != (n_obs,):
        raise ValueError(
            f"discount_factors must have shape ({n_obs},), got {discount_factors.shape}"
        )

    pv = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    missed = np.zeros(n_paths, dtype=np.int64)
    breached = np.zeros(n_paths, dtype=bool)

    for i in range(n_obs):
        perf_i = performance[:, i]

        # --- 1. contingent coupon -------------------------------------------
        pays = alive & (perf_i >= spec.coupon_barrier)
        if spec.memory:
            n_periods = np.where(pays, missed + 1, 0)
            pv += n_periods * spec.coupon_rate * discount_factors[i]
            missed = np.where(pays, 0, np.where(alive, missed + 1, missed))
        else:
            pv += pays * spec.coupon_rate * discount_factors[i]

        # --- barrier monitoring on scheduled dates --------------------------
        if not spec.knock_in_at_maturity:
            breached |= alive & (perf_i < spec.knock_in_barrier)

        # --- 2. automatic call ----------------------------------------------
        is_last = i == n_obs - 1
        level = spec.autocall_levels[i]
        callable_now = not np.isnan(level)
        if callable_now and not is_last:
            called = alive & (perf_i >= level)
            pv += called * 1.0 * discount_factors[i]
            alive &= ~called

    # --- 3. maturity --------------------------------------------------------
    final_perf = performance[:, -1]
    final_level = spec.autocall_levels[-1]
    par_by_call = (
        (final_perf >= final_level)
        if not np.isnan(final_level)
        else np.zeros(n_paths, dtype=bool)
    )
    if spec.knock_in_at_maturity:
        protected = final_perf >= spec.knock_in_barrier
    else:
        # A breach costs principal only if the final level is also below the
        # initial level ("a Knock-In Event has occurred and the Final Level is
        # less than its Initial Level").
        protected = (~breached & (final_perf >= spec.knock_in_barrier)) | (final_perf >= 1.0)

    downside = 1.0 - spec.downside_participation * (1.0 - final_perf)
    redemption = np.where(par_by_call | protected, 1.0, np.maximum(downside, 0.0))
    pv += alive * redemption * discount_factors[-1]

    return pv
