"""Business rules whose behaviour depends on a choice the paper has to state."""

from datetime import timedelta

from src.validate import validate_terms
from tests.test_schema import PRICING, terms


def rules(report):
    return {f.rule for f in report.findings}


def test_initial_level_is_checked_against_the_strike_date_close():
    """Regression test for the TD strike-dated note.

    The filed initial levels are strike-date closes.  Comparing them with the
    pricing-date close flagged a correct extraction as 1.6% off.
    """
    strike = PRICING - timedelta(days=1)
    t = terms(strike_date=strike)  # initial level 100.0
    strike_close = {"^GSPC": 100.0}
    assert "initial_level_matches_close" not in rules(
        validate_terms(t, initial_level_closes=strike_close)
    )
    # A real extraction error is still caught.
    assert "initial_level_matches_close" in rules(
        validate_terms(t, initial_level_closes={"^GSPC": 101.6})
    )


def test_strike_date_after_pricing_date_is_a_hard_error():
    report = validate_terms(terms(strike_date=PRICING + timedelta(days=1)))
    assert "strike_not_after_pricing" in {f.rule for f in report.hard}


def test_nyse_calendar_known_dates():
    from datetime import date

    from src.market.calendar import add_trading_days, is_trading_day, nyse_holidays

    h2026 = nyse_holidays(2026)
    assert date(2026, 4, 3) in h2026  # Good Friday 2026
    assert date(2026, 7, 3) in h2026  # Independence Day observed (Sat 4 Jul)
    assert date(2026, 11, 26) in h2026  # Thanksgiving
    assert date(2027, 12, 31) not in nyse_holidays(2027)  # Sat New Year not observed
    assert not is_trading_day(date(2026, 10, 10))  # Saturday
    # Thu 2 Jul 2026 + 1 trading day skips the observed holiday and the weekend.
    assert add_trading_days(date(2026, 7, 2), 1) == date(2026, 7, 6)
