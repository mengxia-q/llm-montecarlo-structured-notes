"""Engine-level tests: closed-form cross-checks of the simulator, and the
common-random-number property that the price-deviation metric depends on.

The closed-form checks work by degenerating the payoff until an exact value is
available:

* knock-in barrier at 0%  -> redemption is always par  -> price = discount factor
  (no Monte Carlo error at all).
* autocall level at 0%    -> certain call on the first callable date.
* knock-in barrier at 1000% -> redemption is always ``notional * P_N``, so the
  price is the discounted risk-neutral expectation of the performance ratio,
  i.e. ``(spot / initial_level) * exp(-q T)``.  This is the test that would catch
  a wrong risk-neutral drift.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.pricing import CommonRandomNumbers, MarketInputs, ZeroCurve, price_note
from src.pricing.engine import UnsupportedProductError, simulate_performance
from src.schema import Autocall, Coupon, KnockIn, NoteTerms, Underlying

PRICING_DATE = date(2026, 3, 16)
FLAT_RATE = 0.04
DIV = 0.015
VOL = 0.20


def quarterly_dates(n, start=PRICING_DATE, lag_days=3):
    """``n`` quarterly observation dates and their payment dates."""
    obs = [start + timedelta(days=91 * (i + 1)) for i in range(n)]
    pay = [d + timedelta(days=lag_days) for d in obs]
    return obs, pay


def make_terms(
    n_obs=12,
    first_callable=3,
    autocall_pct=None,
    coupon_pct=2.25,
    coupon_barrier=70.0,
    memory=True,
    knock_in=60.0,
    monitoring="at_maturity",
    initial_levels=(100.0,),
    notional=1000.0,
    pay_lag=3,
):
    obs, pay = quarterly_dates(n_obs, lag_days=pay_lag)
    n_callable = n_obs - first_callable + 1
    levels = [100.0] * n_callable if autocall_pct is None else list(autocall_pct)
    n_assets = len(initial_levels)
    return NoteTerms(
        issuer="Test Issuer",
        notional_per_unit=notional,
        currency="USD",
        pricing_date=PRICING_DATE,
        maturity_date=pay[-1],
        underlyings=[
            Underlying(name=f"Index {j}", ticker=f"IX{j}", initial_level=lvl)
            for j, lvl in enumerate(initial_levels)
        ],
        performance_type="single" if n_assets == 1 else "worst_of",
        observation_dates=obs,
        payment_dates=pay,
        autocall=Autocall(first_observation_index=first_callable, levels_pct=levels),
        coupon=Coupon(
            rate_per_period_pct=coupon_pct, barrier_pct=coupon_barrier, memory=memory
        ),
        knock_in=KnockIn(barrier_pct=knock_in, monitoring=monitoring),
    )


def make_market(n_assets=1, spots=None, vol=VOL, div=DIV, rate=FLAT_RATE, rho=0.5):
    spots = np.full(n_assets, 100.0) if spots is None else np.asarray(spots, float)
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return MarketInputs(
        tickers=tuple(f"IX{j}" for j in range(n_assets)),
        spots=spots,
        vols=np.full(n_assets, vol),
        corr=corr,
        div_yields=np.full(n_assets, div),
        curve=ZeroCurve.flat(rate),
    )


# --------------------------------------------------------------------------
# Closed-form cross-checks
# --------------------------------------------------------------------------
def test_zero_knock_in_barrier_reduces_to_a_risk_free_zero_coupon_bond():
    """Always-par redemption, no coupon, never callable -> exactly the discount factor."""
    terms = make_terms(
        n_obs=8, first_callable=8, autocall_pct=[1000.0], coupon_pct=0.0,
        coupon_barrier=1000.0, memory=False, knock_in=0.0,
    )
    result = price_note(terms, make_market(), CommonRandomNumbers(seed=1, n_paths=2_048))
    t_pay = (terms.payment_dates[-1] - PRICING_DATE).days / 365.0
    expected = 1000.0 * np.exp(-FLAT_RATE * t_pay)
    assert result.price_per_1000 == pytest.approx(expected, rel=1e-12)
    assert result.std_error_per_1000 == pytest.approx(0.0, abs=1e-12)


def test_zero_autocall_level_gives_a_certain_call_on_the_first_callable_date():
    terms = make_terms(
        n_obs=8, first_callable=3, autocall_pct=[0.0] * 6, coupon_pct=2.0,
        coupon_barrier=0.0, memory=False, knock_in=0.0,
    )
    result = price_note(terms, make_market(), CommonRandomNumbers(seed=2, n_paths=2_048))
    # Coupons on observations 1..3, then par on the third payment date.
    pay_times = np.array(
        [(d - PRICING_DATE).days / 365.0 for d in terms.payment_dates[:3]]
    )
    dfs = np.exp(-FLAT_RATE * pay_times)
    expected = 1000.0 * (0.02 * dfs.sum() + dfs[-1])
    assert result.price_per_1000 == pytest.approx(expected, rel=1e-12)


def test_pure_performance_payoff_matches_the_risk_neutral_expectation():
    """Validates the GBM drift: price must equal (spot/initial) * exp(-q T)."""
    terms = make_terms(
        n_obs=4, first_callable=4, autocall_pct=[1000.0], coupon_pct=0.0,
        coupon_barrier=1000.0, memory=False, knock_in=1000.0, pay_lag=0,
    )
    result = price_note(
        terms, make_market(), CommonRandomNumbers(seed=3, n_paths=262_144)
    )
    t_final = (terms.observation_dates[-1] - PRICING_DATE).days / 365.0
    expected = 1000.0 * np.exp(-DIV * t_final)
    assert result.price_per_1000 == pytest.approx(expected, rel=2e-3)


def test_mis_extracted_initial_level_scales_the_performance_payoff():
    """A 10% too-high initial level must scale a pure-performance payoff by 1/1.1.

    This is the test that pins down the design decision in
    ``simulate_performance``: paths start at the market close and are divided by
    the *extracted* initial level, so initial_level is price-relevant.
    """
    kwargs = dict(
        n_obs=4, first_callable=4, autocall_pct=[1000.0], coupon_pct=0.0,
        coupon_barrier=1000.0, memory=False, knock_in=1000.0, pay_lag=0,
    )
    crn = CommonRandomNumbers(seed=4, n_paths=131_072)
    market = make_market(spots=[100.0])
    correct = price_note(make_terms(initial_levels=(100.0,), **kwargs), market, crn)
    wrong = price_note(make_terms(initial_levels=(110.0,), **kwargs), market, crn)
    assert wrong.price_per_1000 == pytest.approx(correct.price_per_1000 / 1.1, rel=1e-12)


# --------------------------------------------------------------------------
# Common random numbers: the property the Delta metric relies on
# --------------------------------------------------------------------------
def test_identical_parameters_give_exactly_zero_deviation():
    terms = make_terms()
    crn = CommonRandomNumbers(seed=11, n_paths=16_384)
    market = make_market()
    a = price_note(terms, market, crn)
    b = price_note(make_terms(), market, crn)
    assert a.price_per_1000 == b.price_per_1000  # bit-for-bit, not approx


def test_random_numbers_are_stable_when_the_date_count_changes():
    """Extra trailing observation dates must not disturb the shared prefix."""
    market = make_market()
    crn = CommonRandomNumbers(seed=12, n_paths=4_096)
    short = simulate_performance(make_terms(n_obs=8), market, crn)
    long = simulate_performance(make_terms(n_obs=12), market, crn)
    np.testing.assert_allclose(short, long[:, :8], rtol=0, atol=0)


def test_an_extraction_error_moves_the_price():
    """Sanity floor for the sensitivity analysis: a wrong coupon barrier must change the valuation."""
    market = make_market()
    crn = CommonRandomNumbers(seed=13, n_paths=65_536)
    base = price_note(make_terms(coupon_barrier=70.0), market, crn)
    wrong = price_note(make_terms(coupon_barrier=80.0), market, crn)
    assert abs(base.price_per_1000 - wrong.price_per_1000) > 1.0


# --------------------------------------------------------------------------
# Monotonicity sanity checks
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,low,high,expect_increasing",
    [
        ("coupon_pct", 1.0, 3.0, True),        # more coupon -> worth more
        ("coupon_barrier", 60.0, 85.0, False), # harder to earn -> worth less
        ("knock_in", 50.0, 75.0, False),       # less protection -> worth less
    ],
)
def test_price_monotonicity(field, low, high, expect_increasing):
    market = make_market()
    crn = CommonRandomNumbers(seed=14, n_paths=65_536)
    p_low = price_note(make_terms(**{field: low}), market, crn).price_per_1000
    p_high = price_note(make_terms(**{field: high}), market, crn).price_per_1000
    assert (p_high > p_low) is expect_increasing


def test_worst_of_basket_is_worth_no_more_than_its_weakest_single_underlying():
    crn = CommonRandomNumbers(seed=15, n_paths=65_536)
    single = price_note(
        make_terms(initial_levels=(100.0,)), make_market(1), crn
    ).price_per_1000
    basket = price_note(
        make_terms(initial_levels=(100.0, 100.0, 100.0)), make_market(3, rho=0.5), crn
    ).price_per_1000
    assert basket < single


def test_lower_correlation_lowers_a_worst_of_note():
    crn = CommonRandomNumbers(seed=16, n_paths=65_536)
    terms = make_terms(initial_levels=(100.0, 100.0))
    high = price_note(terms, make_market(2, rho=0.9), crn).price_per_1000
    low = price_note(terms, make_market(2, rho=0.1), crn).price_per_1000
    assert low < high


# --------------------------------------------------------------------------
# Coverage boundaries
# --------------------------------------------------------------------------
def test_daily_barrier_monitoring_is_rejected_rather_than_approximated():
    terms = make_terms(monitoring="continuous_daily")
    with pytest.raises(UnsupportedProductError, match="continuous/daily"):
        price_note(terms, make_market(), CommonRandomNumbers(seed=17, n_paths=1_024))


def test_market_inputs_must_match_the_number_of_underlyings():
    terms = make_terms(initial_levels=(100.0, 100.0))
    with pytest.raises(ValueError, match="market inputs cover"):
        price_note(terms, make_market(1), CommonRandomNumbers(seed=18, n_paths=1_024))


def test_antithetic_reduces_error_on_a_payoff_that_is_monotone_in_the_driver():
    """Exercises the antithetic pair-average standard error.

    Antithetic sampling only helps when the payoff is close to monotone in the
    Brownian driver.  The pure-performance payoff is, so this test pins down that
    the machinery and its pair-average standard error work.  The autocallable
    payoff is *not* monotone and antithetic sampling measurably hurts there,
    which is why it is no longer the default - see ``src.pricing.rng``.
    """
    terms = make_terms(
        n_obs=4, first_callable=4, autocall_pct=[1000.0], coupon_pct=0.0,
        coupon_barrier=1000.0, memory=False, knock_in=1000.0, pay_lag=0,
    )
    market = make_market()
    plain = price_note(
        terms, market,
        CommonRandomNumbers(seed=19, n_paths=32_768, generator="iid"),
    )
    anti = price_note(
        terms, market,
        CommonRandomNumbers(seed=19, n_paths=32_768, generator="iid", antithetic=True),
    )
    assert anti.std_error_per_1000 < 0.5 * plain.std_error_per_1000


def test_antithetic_requires_the_iid_generator():
    with pytest.raises(ValueError, match="antithetic variates require"):
        CommonRandomNumbers(seed=1, n_paths=1_024, generator="sobol", antithetic=True)


def test_sobol_beats_iid_on_the_autocallable_payoff():
    """Single-seed smoke check of the Sobol variance-reduction claim."""
    terms = make_terms()
    market = make_market()
    sobol = price_note(terms, market, CommonRandomNumbers(seed=20, n_paths=16_384))
    iid = price_note(
        terms, market, CommonRandomNumbers(seed=20, n_paths=16_384, generator="iid")
    )
    # Prices must agree to within a few standard errors ...
    assert abs(sobol.price_per_1000 - iid.price_per_1000) < 5 * iid.std_error_per_1000
    # ... and the Sobol estimate must not be the noisier of the two.
    assert sobol.std_error_per_1000 <= iid.std_error_per_1000 * 1.05


# --------------------------------------------------------------------------
# Chunked simulation (convergence study runs to 2^20 paths)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "generator,antithetic", [("sobol", False), ("iid", False), ("iid", True)]
)
def test_chunked_pricing_matches_single_block(generator, antithetic):
    """Chunks are consecutive rows of one stream, so the estimate is unchanged."""
    terms = make_terms(initial_levels=(100.0, 100.0))
    market = make_market(n_assets=2)
    crn = CommonRandomNumbers(
        seed=7, n_paths=4_096, generator=generator, antithetic=antithetic,
        max_steps=12, max_assets=2,
    )
    whole = price_note(terms, market, crn)
    chunked = price_note(terms, market, crn, chunk_paths=1_024)
    assert chunked.price_per_1000 == pytest.approx(whole.price_per_1000, rel=1e-12)
    assert chunked.std_error_per_1000 == pytest.approx(
        whole.std_error_per_1000, rel=1e-9
    )


def test_chunks_concatenate_to_the_fixed_block():
    crn = CommonRandomNumbers(seed=3, n_paths=2_048, max_steps=8, max_assets=3)
    chunks = list(crn.iter_draws(6, 2, 512))
    assert len(chunks) == 4
    np.testing.assert_array_equal(np.concatenate(chunks), crn.draw(6, 2))
