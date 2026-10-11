"""Risk-free curve from the FRED constant-maturity Treasury series.

The daily CMT series are free and require no API key.  Two approximations are
made, and both are stated in the paper:

1. Par yields are used directly as zero rates.  A full bootstrap would change
   the level slightly; it is applied identically to both valuations of a filing
   and therefore cancels in the price-deviation measurement.
2. Quoted yields are treated as semiannually compounded and converted to a
   continuously compounded basis via ``r = 2 ln(1 + y/2)``.
"""

from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..pricing.market import ZeroCurve

class CurveDataError(RuntimeError):
    """Raised when the risk-free curve cannot be built."""


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: FRED occasionally stalls on the first request; retry with backoff rather than
#: failing a long experiment run.
_RETRY = dict(attempts=4, timeout=120)


def _get_with_retry(url: str, params: dict) -> requests.Response:
    last: Exception | None = None
    for attempt in range(_RETRY["attempts"]):
        try:
            response = requests.get(
                url,
                params=params,
                headers={
                    "User-Agent": "structured-notes-llm research",
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=_RETRY["timeout"],
            )
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last = exc
            if attempt < _RETRY["attempts"] - 1:
                time.sleep(2.0 * (attempt + 1))
    raise CurveDataError(f"could not download {url}: {last}") from last

#: FRED series id -> tenor in years.
CMT_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
}

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "market" / "treasury_cmt.csv"


def fetch_cmt(start: date, end: date, cache: Path | None = DEFAULT_CACHE) -> pd.DataFrame:
    """Daily CMT yields in percent, indexed by observation date.

    ``fredgraph.csv`` ignores the ``cosd``/``coed`` window parameters and always
    returns each series in full (about 800 kB for these nine, back to 1962), so
    the whole history is downloaded once and cached; every later call is served
    from disk.  ``start`` and ``end`` are used only to decide whether the cache
    is still current.
    """
    cached: pd.DataFrame | None = None
    if cache is not None and cache.exists():
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if not cached.empty and cached.index.min().date() <= start and cached.index.max().date() >= end:
            return cached

    response = _get_with_retry(
        FRED_CSV, params={"id": ",".join(CMT_SERIES)}
    )
    frame = pd.read_csv(io.StringIO(response.text))
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").apply(pd.to_numeric, errors="coerce")

    if cached is not None and not cached.empty:
        frame = pd.concat([cached, frame])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache)
    return frame


def load_zero_curve(
    as_of: date,
    *,
    cache: Path | None = DEFAULT_CACHE,
    max_staleness_days: int = 7,
) -> ZeroCurve:
    """Continuously compounded zero curve on the last business day <= ``as_of``."""
    frame = fetch_cmt(as_of - timedelta(days=30), as_of, cache=cache)
    window = frame[frame.index.date <= as_of].dropna(how="all")
    if window.empty:
        raise CurveDataError(f"no Treasury quotes available on or before {as_of}")

    row = window.iloc[-1]
    quote_date = window.index[-1].date()
    staleness = (as_of - quote_date).days
    if staleness > max_staleness_days:
        raise CurveDataError(
            f"latest Treasury quote is {quote_date}, {staleness} days before the "
            f"requested {as_of}; refresh the cache"
        )

    tenors, rates = [], []
    for series, tenor in CMT_SERIES.items():
        value = row.get(series)
        if value is None or pd.isna(value):
            continue
        par = float(value) / 100.0
        tenors.append(tenor)
        rates.append(2.0 * np.log1p(par / 2.0))

    if len(tenors) < 2:
        raise CurveDataError(f"only {len(tenors)} usable tenors on {quote_date}")

    order = np.argsort(tenors)
    return ZeroCurve(
        tenors=tuple(float(tenors[i]) for i in order),
        rates=tuple(float(rates[i]) for i in order),
    )
