"""Deterministic-path unit tests for the payoff rules.

Every test drives ``discounted_cashflows`` with a hand-constructed performance
matrix so that the expected cash flows can be written down by hand, with no
Monte Carlo noise involved.  Discount factors are set to 1.0 unless the test is
specifically about discounting.
"""

import numpy as np
import pytest

from src.pricing.payoff import PayoffSpec, discounted_cashflows

NAN = np.nan


def spec(
    coupon_rate=0.025,
    coupon_barrier=0.70,
    memory=False,
    autocall_levels=(1.0, 1.0, 1.0, 1.0),
    knock_in_barrier=0.60,
    knock_in_at_maturity=True,
    downside_participation=1.0,
):
    return PayoffSpec(
        coupon_rate=coupon_rate,
        coupon_barrier=coupon_barrier,
        memory=memory,
        autocall_levels=np.array(autocall_levels, dtype=float),
        knock_in_barrier=knock_in_barrier,
        knock_in_at_maturity=knock_in_at_maturity,
        downside_participation=downside_participation,
    )


def pv(perf, s, dfs=None):
    perf = np.atleast_2d(np.asarray(perf, dtype=float))
    dfs = np.ones(perf.shape[1]) if dfs is None else np.asarray(dfs, dtype=float)
    return discounted_cashflows(perf, s, dfs)[0]


# --------------------------------------------------------------------------
# Automatic call
# --------------------------------------------------------------------------
def test_calls_on_first_callable_date_and_pays_that_coupon():
    """Rising path: not callable at i=0, called at i=1 -> 2 coupons + par."""
    s = spec(autocall_levels=(NAN, 1.00, 1.00, 1.00))
    assert pv([1.05, 1.10, 1.20, 1.30], s) == pytest.approx(1.0 + 2 * 0.025)


def test_not_called_before_first_callable_date():
    """A path above the call level at i=0 must not terminate when i=0 is not callable."""
    s = spec(autocall_levels=(NAN, NAN, 1.00, 1.00))
    # Dips below the call level at i=1..2, recovers at i=3 -> called at maturity leg.
    perf = [1.50, 0.95, 0.95, 1.50]
    # 4 coupons (all above 70%) + par at maturity.
    assert pv(perf, s) == pytest.approx(1.0 + 4 * 0.025)


def test_step_down_autocall_calls_later_level():
    """Performance of 96% calls only once the schedule steps down to 95%."""
    s = spec(autocall_levels=(1.00, 1.00, 0.95, 0.95))
    perf = [0.96, 0.96, 0.96, 0.96]
    # Coupons at i=0,1 then called at i=2 with par: 3 coupons + par.
    assert pv(perf, s) == pytest.approx(1.0 + 3 * 0.025)


def test_call_on_final_observation_pays_par_not_downside():
    s = spec(autocall_levels=(NAN, 1.00, 1.00, 1.00))
    assert pv([0.5, 0.5, 0.5, 1.05], s) == pytest.approx(1.0 + 0.025)


# --------------------------------------------------------------------------
# Contingent coupon and the memory feature
# --------------------------------------------------------------------------
def test_coupon_skipped_below_barrier():
    s = spec(autocall_levels=(NAN, NAN, NAN, NAN))
    # Below the 70% coupon barrier at i=1,2; above at i=0,3. Ends above knock-in.
    perf = [0.80, 0.65, 0.65, 0.80]
    assert pv(perf, s) == pytest.approx(1.0 + 2 * 0.025)


def test_memory_coupon_pays_arrears_on_recovery():
    s = spec(memory=True, autocall_levels=(NAN, NAN, NAN, NAN))
    perf = [0.80, 0.65, 0.65, 0.80]
    # i=0 pays 1; i=1,2 missed; i=3 pays 3 (itself + 2 arrears) -> 4 coupons.
    assert pv(perf, s) == pytest.approx(1.0 + 4 * 0.025)


def test_memory_never_recovering_pays_only_early_coupons():
    s = spec(memory=True, autocall_levels=(NAN, NAN, NAN, NAN))
    perf = [0.80, 0.65, 0.65, 0.50]
    # Only i=0 pays; maturity below the 60% knock-in -> downside of 0.50.
    assert pv(perf, s) == pytest.approx(0.50 + 0.025)


def test_memory_arrears_stop_accruing_after_call():
    """Once called, no further coupon or arrears may be paid."""
    s = spec(memory=True, autocall_levels=(NAN, 1.00, 1.00, 1.00))
    perf = [0.65, 1.10, 1.10, 1.10]
    # i=0 missed; i=1 pays 2 (itself + 1 arrear) and calls -> nothing afterwards.
    assert pv(perf, s) == pytest.approx(1.0 + 2 * 0.025)


def test_memory_and_non_memory_agree_when_no_coupon_is_missed():
    perf = [0.95, 0.95, 0.95, 0.95]
    s_mem = spec(memory=True, autocall_levels=(NAN, NAN, NAN, NAN))
    s_plain = spec(memory=False, autocall_levels=(NAN, NAN, NAN, NAN))
    assert pv(perf, s_mem) == pytest.approx(pv(perf, s_plain))


# --------------------------------------------------------------------------
# Maturity redemption
# --------------------------------------------------------------------------
def test_par_when_final_above_knock_in():
    s = spec(autocall_levels=(NAN, NAN, NAN, NAN))
    # Final 65% is below the 70% coupon barrier but above the 60% knock-in.
    assert pv([0.65, 0.65, 0.65, 0.65], s) == pytest.approx(1.0)


def test_downside_when_final_below_knock_in():
    s = spec(autocall_levels=(NAN, NAN, NAN, NAN))
    assert pv([0.55, 0.55, 0.55, 0.55], s) == pytest.approx(0.55)


def test_total_loss_is_floored_at_zero():
    s = spec(autocall_levels=(NAN, NAN, NAN, NAN), downside_participation=2.0)
    # 2x participation on a 60% decline would give -0.2 without the floor.
    assert pv([0.40, 0.40, 0.40, 0.40], s) == pytest.approx(0.0)


def test_european_vs_observation_date_knock_in_monitoring():
    """A path that dips below the barrier mid-life but recovers above it."""
    perf = [0.95, 0.55, 0.95, 0.95]
    euro = spec(autocall_levels=(NAN, NAN, NAN, NAN), knock_in_at_maturity=True)
    amer = spec(autocall_levels=(NAN, NAN, NAN, NAN), knock_in_at_maturity=False)
    coupons = 3 * 0.025  # i=1 is below the 70% coupon barrier
    assert pv(perf, euro) == pytest.approx(1.0 + coupons)
    # With observation-date monitoring the barrier is permanently breached, so the
    # final 95% performance is paid instead of par.
    assert pv(perf, amer) == pytest.approx(0.95 + coupons)


def test_breached_barrier_repays_only_par_when_final_is_above_initial():
    """After a knock-in event, a final level above the initial level repays par."""
    s = spec(autocall_levels=(NAN, NAN, NAN, NAN), knock_in_at_maturity=False)
    coupons = 3 * 0.025  # i=1 is below the 70% coupon barrier
    assert pv([0.95, 0.55, 0.95, 1.20], s) == pytest.approx(1.0 + coupons)


# --------------------------------------------------------------------------
# Discounting and vectorisation
# --------------------------------------------------------------------------
def test_cashflows_are_discounted_to_their_payment_dates():
    s = spec(autocall_levels=(NAN, 1.00))
    dfs = np.array([0.99, 0.98])
    # Coupon at i=0 discounted at 0.99; coupon + par at i=1 discounted at 0.98.
    expected = 0.025 * 0.99 + (1.0 + 0.025) * 0.98
    assert pv([0.80, 1.10], s, dfs) == pytest.approx(expected)


def test_paths_are_independent_under_vectorisation():
    s = spec(memory=True, autocall_levels=(NAN, 1.00, 1.00, 1.00))
    perf = np.array(
        [
            [1.05, 1.10, 1.20, 1.30],  # called at i=1
            [0.80, 0.65, 0.65, 0.80],  # memory arrears at i=3
            [0.55, 0.55, 0.55, 0.55],  # downside
        ]
    )
    got = discounted_cashflows(perf, s, np.ones(4))
    want = [
        1.0 + 2 * 0.025,
        1.0 + 4 * 0.025,
        0.55 + 0.0,
    ]
    np.testing.assert_allclose(got, want)


def test_shape_mismatches_are_rejected():
    s = spec(autocall_levels=(NAN, NAN, NAN, NAN))
    with pytest.raises(ValueError, match="autocall_levels must have shape"):
        discounted_cashflows(np.ones((2, 3)), s, np.ones(3))
    with pytest.raises(ValueError, match="discount_factors must have shape"):
        discounted_cashflows(np.ones((2, 4)), s, np.ones(3))
