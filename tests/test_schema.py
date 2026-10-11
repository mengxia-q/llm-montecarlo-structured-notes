"""Structural invariants of ``NoteTerms``, and the autocall-window convention."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.schema import Autocall, Coupon, KnockIn, NoteTerms, Underlying

PRICING = date(2026, 3, 16)


def terms(**overrides):
    n_obs = overrides.pop("n_obs", 12)
    first = overrides.pop("first_observation_index", 3)
    levels = overrides.pop("levels_pct", None)
    obs = [PRICING + timedelta(days=91 * (i + 1)) for i in range(n_obs)]
    pay = [d + timedelta(days=3) for d in obs]
    if levels is None:
        levels = [100.0] * (n_obs - first + 1)
    payload = dict(
        issuer="Test",
        notional_per_unit=1000.0,
        pricing_date=PRICING,
        maturity_date=pay[-1],
        underlyings=[Underlying(name="S&P 500 Index", ticker="SPX", initial_level=100.0)],
        performance_type="single",
        observation_dates=obs,
        payment_dates=pay,
        autocall=Autocall(first_observation_index=first, levels_pct=levels),
        coupon=Coupon(rate_per_period_pct=2.25, barrier_pct=70.0, memory=True),
        knock_in=KnockIn(barrier_pct=60.0, monitoring="at_maturity"),
    )
    payload.update(overrides)
    return NoteTerms(**payload)


def test_callable_window_may_stop_before_the_final_observation():
    """Filings commonly make a note callable on interim valuation dates only.

    Regression test: an earlier invariant required the callable window to reach
    the last observation date, which rejected the majority of real Citigroup and
    Goldman filings even though the extraction was correct.
    """
    interim_only = terms(n_obs=12, first_observation_index=3, levels_pct=[100.0] * 9)
    mask = interim_only.callable_mask()
    assert mask[:2] == [False, False]
    assert all(mask[2:11])
    assert mask[11] is False
    assert interim_only.autocall_level_at(11) is None
    assert interim_only.autocall_level_at(2) == 100.0


def test_callable_window_may_reach_the_final_observation():
    fully = terms(n_obs=12, first_observation_index=3, levels_pct=[100.0] * 10)
    assert fully.callable_mask()[11] is True
    assert fully.autocall_level_at(11) == 100.0


def test_callable_window_may_not_run_past_the_schedule():
    with pytest.raises(ValidationError, match="runs past the 12 observation dates"):
        terms(n_obs=12, first_observation_index=3, levels_pct=[100.0] * 11)


def test_step_down_ladder_is_read_in_order():
    stepped = terms(n_obs=6, first_observation_index=2, levels_pct=[100.0, 100.0, 95.0, 95.0, 90.0])
    assert [stepped.autocall_level_at(i) for i in range(6)] == [
        None, 100.0, 100.0, 95.0, 95.0, 90.0
    ]


def test_payment_and_observation_schedules_must_align():
    with pytest.raises(ValidationError, match="must have equal length"):
        obs = [PRICING + timedelta(days=91 * (i + 1)) for i in range(4)]
        terms(n_obs=4, observation_dates=obs, payment_dates=obs[:3], levels_pct=[100.0, 100.0])


def test_first_callable_index_must_be_within_the_schedule():
    with pytest.raises(ValidationError, match="exceeds the number of observation dates"):
        terms(n_obs=4, first_observation_index=5, levels_pct=[100.0])


def test_performance_type_must_match_the_basket_size():
    with pytest.raises(ValidationError, match="requires at least two underlyings"):
        terms(performance_type="worst_of")
    with pytest.raises(ValidationError, match="requires exactly one underlying"):
        terms(
            performance_type="single",
            underlyings=[
                Underlying(name="S&P 500 Index", initial_level=100.0),
                Underlying(name="Nasdaq-100 Index", initial_level=200.0),
            ],
        )


def test_final_valuation_date_is_the_last_observation():
    t = terms(n_obs=5, first_observation_index=2, levels_pct=[100.0] * 4)
    assert t.final_valuation_date == t.observation_dates[-1]
    assert t.n_observations == 5


def test_initial_levels_are_dated_by_the_strike_date_when_there_is_one():
    assert terms().initial_level_date == PRICING
    struck = terms(strike_date=PRICING - timedelta(days=1))
    assert struck.initial_level_date == PRICING - timedelta(days=1)
