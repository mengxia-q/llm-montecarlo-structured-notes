"""Price history, volatility, correlation and dividend yields.

Estimation choices, all stated in the paper's Pricing Template section:

* Volatility: annualised standard deviation of the daily log returns over the
  ``ESTIMATION_WINDOW`` (252) trading days ending on the pricing date.
* Correlation: Pearson correlation of the same log returns, computed on dates
  where *every* underlying traded, so the matrix is internally consistent.
* Dividend yield: constant per underlying, from the trailing twelve months of
  distributions of the underlying (for an ETF) or of its proxy ETF (for an
  index, whose published level is price-return and excludes dividends).

All series are cached under ``data/market/`` so that a re-run of the experiment
uses byte-identical market data; this is what lets the paired valuations differ
only in the extracted parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..pricing.market import MarketInputs
from ..schema import NoteTerms
from .curve import load_zero_curve
from .tickers import TickerInfo, resolve_underlying

#: Trading days of daily log returns used to estimate volatility and correlation.
ESTIMATION_WINDOW = 252

TRADING_DAYS_PER_YEAR = 252.0

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "market"


class MarketDataError(RuntimeError):
    """Raised when market data is missing or too sparse to estimate inputs."""


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace("^", "IDX_").replace(".", "_").replace("/", "_")
    return cache_dir / f"{safe}.csv"


def price_history(
    ticker: str,
    start: date,
    end: date,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    kind: str = "close",
) -> pd.Series:
    """Daily close (or dividend) series for ``ticker``, cached on disk.

    ``kind="close"`` returns unadjusted closes, which is what a filing's
    "Starting Value" refers to.  ``kind="dividends"`` returns the distribution
    series used for the dividend-yield estimate.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(f"{ticker}_{kind}", cache_dir)

    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if not cached.empty and cached.index.min().date() <= start and cached.index.max().date() >= end:
            return cached

    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handle = yf.Ticker(ticker)
        if kind == "dividends":
            series = handle.dividends
        else:
            # Fetch a generous window so the cache serves later requests too.
            frame = handle.history(
                start=(start - timedelta(days=400)).isoformat(),
                end=(end + timedelta(days=5)).isoformat(),
                auto_adjust=False,
            )
            series = frame["Close"] if not frame.empty else pd.Series(dtype=float)

    if series.empty:
        raise MarketDataError(f"no {kind} data returned for {ticker!r}")

    series = series.copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = kind
    series.to_csv(path)
    return series


def _close_on_or_before(series: pd.Series, as_of: date, ticker: str) -> tuple[float, date]:
    window = series[series.index.date <= as_of]
    if window.empty:
        raise MarketDataError(f"no {ticker} close on or before {as_of}")
    return float(window.iloc[-1]), window.index[-1].date()


def dividend_yield(
    info: TickerInfo, as_of: date, *, cache_dir: Path = DEFAULT_CACHE_DIR
) -> float:
    """Trailing twelve-month distribution yield of the dividend proxy."""
    try:
        divs = price_history(
            info.dividend_proxy, as_of - timedelta(days=400), as_of,
            cache_dir=cache_dir, kind="dividends",
        )
        closes = price_history(
            info.dividend_proxy, as_of - timedelta(days=30), as_of, cache_dir=cache_dir
        )
    except MarketDataError:
        raise MarketDataError(
            f"cannot estimate a dividend yield for {info.canonical_name} "
            f"(proxy {info.dividend_proxy})"
        )

    price, _ = _close_on_or_before(closes, as_of, info.dividend_proxy)
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=365)
    recent = divs[(divs.index > cutoff) & (divs.index <= pd.Timestamp(as_of))]
    if recent.empty:
        return 0.0
    return float(recent.sum() / price)


@dataclass(frozen=True)
class Estimates:
    vols: np.ndarray
    corr: np.ndarray
    n_observations: int
    first_return_date: date
    last_return_date: date


def estimate_vol_corr(
    closes: dict[str, pd.Series], as_of: date, window: int = ESTIMATION_WINDOW
) -> Estimates:
    """Annualised volatilities and the correlation matrix of daily log returns.

    Series are intersected on common trading days first, so that the volatility
    and correlation estimates come from exactly the same observations.
    """
    tickers = list(closes)
    frame = pd.DataFrame(
        {t: closes[t][closes[t].index.date <= as_of] for t in tickers}
    ).dropna(how="any")
    if len(frame) < window + 1:
        raise MarketDataError(
            f"only {len(frame)} common trading days available before {as_of}; "
            f"{window + 1} are needed for a {window}-day estimate"
        )

    frame = frame.iloc[-(window + 1):]
    returns = np.log(frame / frame.shift(1)).dropna(how="any")
    vols = returns.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS_PER_YEAR)
    corr = returns.corr().to_numpy() if len(tickers) > 1 else np.ones((1, 1))

    return Estimates(
        vols=vols,
        corr=corr,
        n_observations=len(returns),
        first_return_date=returns.index[0].date(),
        last_return_date=returns.index[-1].date(),
    )


def initial_level_closes(
    terms: NoteTerms, *, cache_dir: Path = DEFAULT_CACHE_DIR
) -> dict[str, float]:
    """Closes on ``terms.initial_level_date``, keyed by market-data ticker.

    This is what the filed initial levels should equal, and differs from the
    pricing-date spot that valuation starts from whenever the note has a
    separate strike date.
    """
    as_of = terms.initial_level_date
    out: dict[str, float] = {}
    for u in terms.underlyings:
        ticker = resolve_underlying(u.name, u.ticker).yahoo_ticker
        series = price_history(ticker, as_of - timedelta(days=10), as_of, cache_dir=cache_dir)
        out[ticker], _ = _close_on_or_before(series, as_of, ticker)
    return out


def build_market_inputs(
    terms: NoteTerms,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    window: int = ESTIMATION_WINDOW,
) -> MarketInputs:
    """Assemble the market state for one filing as of its pricing date.

    The result is a function of the pricing date and the *identity* of the
    underlyings only.  It deliberately does not depend on any other extracted
    field, so the same ``MarketInputs`` object can be reused for both valuations
    of a filing.
    """
    as_of = terms.pricing_date
    infos = [resolve_underlying(u.name, u.ticker) for u in terms.underlyings]
    tickers = [info.yahoo_ticker for info in infos]

    closes = {
        t: price_history(t, as_of - timedelta(days=600), as_of, cache_dir=cache_dir)
        for t in dict.fromkeys(tickers)
    }

    spots, spot_dates = [], []
    for ticker in tickers:
        value, when = _close_on_or_before(closes[ticker], as_of, ticker)
        spots.append(value)
        spot_dates.append(when)

    estimates = estimate_vol_corr({t: closes[t] for t in tickers}, as_of, window=window)
    divs = [dividend_yield(info, as_of, cache_dir=cache_dir) for info in infos]

    return MarketInputs(
        tickers=tuple(tickers),
        spots=np.array(spots, dtype=float),
        vols=np.asarray(estimates.vols, dtype=float),
        corr=np.asarray(estimates.corr, dtype=float),
        div_yields=np.array(divs, dtype=float),
        curve=load_zero_curve(as_of),
        meta={
            "as_of": as_of.isoformat(),
            "canonical_names": [info.canonical_name for info in infos],
            "spot_dates": [d.isoformat() for d in spot_dates],
            "estimation_window": window,
            "n_return_observations": estimates.n_observations,
            "return_window": [
                estimates.first_return_date.isoformat(),
                estimates.last_return_date.isoformat(),
            ],
        },
    )
