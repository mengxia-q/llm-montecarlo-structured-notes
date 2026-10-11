"""NYSE trading calendar, for payment dates that filings give only as a rule.

Some filings never list payment dates and say only "the third business day after
each valuation date".  The labelling guideline is to derive them with this
module rather than by hand, and to mark them as derived.  The rule set is the
NYSE's full-day closures (weekend observance included); special one-off
closures are not modelled.  Getting a payment date wrong by a day or two moves
the price by a few cents at most, because it only enters through discounting.

Usage:
    python -m src.market.calendar 2026-10-06 2027-01-06 --plus 3
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from functools import lru_cache


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> frozenset[date]:
    days = {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),  # Juneteenth
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    # New Year's Day: a Saturday holiday is NOT observed on the prior Friday.
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        days.add(ny + timedelta(days=1))
    elif ny.weekday() < 5:
        days.add(ny)
    return frozenset(days)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def add_trading_days(d: date, n: int) -> date:
    """The ``n``-th NYSE trading day after ``d`` (``n`` >= 1)."""
    out = d
    while n > 0:
        out += timedelta(days=1)
        if is_trading_day(out):
            n -= 1
    return out


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dates", nargs="+", type=date.fromisoformat)
    parser.add_argument("--plus", type=int, required=True, help="business days to add")
    args = parser.parse_args()
    for d in args.dates:
        print(add_trading_days(d, args.plus).isoformat())


if __name__ == "__main__":  # pragma: no cover
    main()
