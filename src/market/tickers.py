"""Mapping from underlying names as written in 424B2 filings to market-data tickers.

Scope decision: the dataset is restricted to equity *index* and *ETF*
underlyings.  Single-stock underlyings are excluded at dataset-construction time
and reported as such, because they would require discrete-dividend handling and
split adjustment that the constant-dividend-yield template does not model.

Each index is paired with a liquid ETF used solely as a dividend-yield proxy:
index price series are price-return (they exclude dividends), so the dividend
yield has to come from somewhere else, and the ETF's trailing twelve-month
distributions divided by its price is a reproducible, free estimate.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


class UnknownUnderlyingError(LookupError):
    """Raised when an underlying name cannot be mapped to a market-data ticker."""


@dataclass(frozen=True)
class TickerInfo:
    canonical_name: str
    yahoo_ticker: str
    dividend_proxy: str
    kind: str  # "index" or "etf"


# Canonical registry.  Keys are normalised names; see ``_normalise``.
_INDEX_REGISTRY: dict[str, TickerInfo] = {
    "sp500": TickerInfo("S&P 500 Index", "^GSPC", "SPY", "index"),
    "nasdaq100": TickerInfo("Nasdaq-100 Index", "^NDX", "QQQ", "index"),
    "russell2000": TickerInfo("Russell 2000 Index", "^RUT", "IWM", "index"),
    "dowjonesindustrialaverage": TickerInfo(
        "Dow Jones Industrial Average", "^DJI", "DIA", "index"
    ),
    "eurostoxx50": TickerInfo("EURO STOXX 50 Index", "^STOXX50E", "FEZ", "index"),
    "nikkei225": TickerInfo("Nikkei 225 Index", "^N225", "EWJ", "index"),
    "sptsx60": TickerInfo("S&P/TSX 60 Index", "^TX60", "XIU.TO", "index"),
    "ftse100": TickerInfo("FTSE 100 Index", "^FTSE", "ISF.L", "index"),
    "swissmarket": TickerInfo("Swiss Market Index", "^SSMI", "EWL", "index"),
    "hangseng": TickerInfo("Hang Seng Index", "^HSI", "EWH", "index"),
    "sp500equalweight": TickerInfo(
        "S&P 500 Equal Weight Index", "^SPXEW", "RSP", "index"
    ),
    "sp400midcap": TickerInfo("S&P MidCap 400 Index", "^SP400", "MDY", "index"),
}

#: ETFs that appear directly as underlyings.  Their own distributions are the
#: dividend source, so they are their own proxy.
_ETF_TICKERS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "EEM", "EFA", "XLF", "XLE", "XLK", "XLV",
    "XBI", "SMH", "KRE", "GDX", "TLT", "HYG", "IYR", "VNQ", "FXI", "EWZ",
    "RSP", "MDY", "ARKK", "SOXX", "IBB", "XOP", "XRT", "XLU", "XLI", "XLP",
)

#: Phrases that identify an underlying as out of scope.  Matched on the
#: normalised name before registry lookup.
_OUT_OF_SCOPE_MARKERS: tuple[tuple[str, str], ...] = (
    ("futuresexcessreturn", "futures excess-return index"),
    ("excessreturn", "excess-return index"),
    ("decrement", "decrement index"),
    ("riskcontrol", "risk-control / volatility-target index"),
    ("volatilitytarget", "risk-control / volatility-target index"),
    ("currencyhedged", "currency-hedged index"),
    ("totalreturn", "total-return index"),
)

# Aliases that normalise to a registry key.
_ALIASES: dict[str, str] = {
    "spx": "sp500",
    "sandp500": "sp500",
    "sandp500index": "sp500",
    "sp500index": "sp500",
    "standardandpoors500index": "sp500",
    "ndx": "nasdaq100",
    "nasdaq100index": "nasdaq100",
    "nasdaq100": "nasdaq100",
    "rty": "russell2000",
    "russell2000index": "russell2000",
    "indu": "dowjonesindustrialaverage",
    "djia": "dowjonesindustrialaverage",
    "dowjonesindustrialaverageindex": "dowjonesindustrialaverage",
    "sx5e": "eurostoxx50",
    "eurostoxx50index": "eurostoxx50",
    "nky": "nikkei225",
    "nikkei225": "nikkei225",
    "nikkeistockaverage": "nikkei225",
    "ukx": "ftse100",
    "smi": "swissmarket",
    "swissmarketindex": "swissmarket",
    "hsi": "hangseng",
    "sptsx60index": "sptsx60",
    "sptx60": "sptsx60",
    "sp500equalweightindex": "sp500equalweight",
    "spw": "sp500equalweight",
    "sandp500equalweightindex": "sp500equalweight",
    "sandp500equalweight": "sp500equalweight",
    "sandpmidcap400index": "sp400midcap",
    "sandpmidcap400": "sp400midcap",
    "spmidcap400index": "sp400midcap",
    "mid": "sp400midcap",
    "sp400": "sp400midcap",
    "spmidcap400index": "sp400midcap",
    "mid": "sp400midcap",
}


#: Normalised ETF *display names* as they appear in filings -> ticker.  The
#: normaliser strips the words "etf", "trust", "shares" and "fund", so these keys
#: are what remains of the marketing name.
_ETF_NAME_ALIASES: dict[str, str] = {
    "spdrsandp500": "SPY",
    "spdrsandp500etftrust": "SPY",
    "invescoqqq": "QQQ",
    "invescoqqqseries1": "QQQ",
    "ishsrussell2000": "IWM",
    "isharesrussell2000": "IWM",
    "spdrdowjonesindustrialaverage": "DIA",
    "isharesmscieemergingmarkets": "EEM",
    "isharesmsciemergingmarkets": "EEM",
    "isharesmscieafe": "EFA",
    "financialselectsectorspdr": "XLF",
    "energyselectsectorspdr": "XLE",
    "technologyselectsectorspdr": "XLK",
    "healthcareselectsectorspdr": "XLV",
    "spdrsandpbiotech": "XBI",
    "vaneckvectorssemiconductor": "SMH",
    "vanecksemiconductor": "SMH",
    "spdrsandpregionalbanking": "KRE",
    "vaneckvectorsgoldminers": "GDX",
    "vaneckgoldminers": "GDX",
    "ishares20yeartreasurybond": "TLT",
    "isharesiboxxhighyieldcorporatebond": "HYG",
    "isharesusrealestate": "IYR",
    "vanguardrealestate": "VNQ",
    "ishareschinalargecap": "FXI",
    "isharesmscibrazil": "EWZ",
    "invescosandp500equalweight": "RSP",
    "spdrsandpmidcap400etftrustseries1": "MDY",
}


def _normalise(name: str) -> str:
    """Strip punctuation, trademark marks, casing and the word 'index'."""
    # Trademark marks must be stripped *before* NFKD: the compatibility
    # decomposition turns "\u2120" into "SM" and "\u2122" into "TM", which would
    # otherwise survive into the normalised key.
    text = re.sub(r"[\u00ae\u2122\u2120\u00a9\u2117]", "", name)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("&", "and")
    text = text.lower()
    text = re.sub(r"\b(the|index|indices|shares|etf|trust|fund)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def resolve_underlying(name: str, ticker_hint: str | None = None) -> TickerInfo:
    """Map a filing's underlying name to market-data tickers.

    ``ticker_hint`` is the ticker the extraction reported, used only as a
    fallback; the name is authoritative because tickers in filings are
    inconsistently formatted.

    Raises
    ------
    UnknownUnderlyingError
        If the underlying is out of scope (single stock, decrement index,
        excess-return index, ...) or simply unrecognised.  Callers are expected
        to exclude the filing and count it, rather than guess.
    """
    key = _normalise(name)

    for marker, description in _OUT_OF_SCOPE_MARKERS:
        if marker in key:
            raise UnknownUnderlyingError(
                f"{name!r} is a {description}, which is outside the dataset scope"
            )

    key = _ALIASES.get(key, key)
    if key in _INDEX_REGISTRY:
        return _INDEX_REGISTRY[key]

    etf = _ETF_NAME_ALIASES.get(key)
    if etf is not None:
        return TickerInfo(etf, etf, etf, "etf")

    for candidate in (ticker_hint or "", name):
        bare = re.sub(r"[^A-Za-z.]", "", candidate).upper()
        if bare in _ETF_TICKERS:
            return TickerInfo(bare, bare, bare, "etf")

    raise UnknownUnderlyingError(
        f"{name!r} is not a registered index or ETF underlying. Single-stock "
        "underlyings are out of scope; add the entry to _INDEX_REGISTRY if the "
        "underlying is an index that belongs in the dataset."
    )


#: Human-readable name fragments for every in-scope underlying, used by the
#: dataset-construction screen in ``src.crawl.screen`` to reject filings linked to
#: out-of-scope underlyings before any extraction call is spent on them.
SCOPE_NAME_PATTERNS: tuple[str, ...] = (
    "s&p 500",
    "sandp 500",
    "nasdaq-100",
    "nasdaq 100",
    "russell 2000",
    "dow jones industrial average",
    "euro stoxx 50",
    "nikkei 225",
    "s&p/tsx 60",
    "ftse 100",
    "swiss market index",
    "hang seng",
    "s&p midcap 400",
    "spdr s&p 500",
    "invesco qqq",
    "ishares russell 2000",
    "ishares msci",
    "select sector spdr",
    "vaneck",
    "spdr s&p biotech",
    "spdr s&p regional banking",
)


def scope_name_patterns() -> tuple[str, ...]:
    """Lower-cased name fragments that identify an in-scope underlying."""
    return SCOPE_NAME_PATTERNS
