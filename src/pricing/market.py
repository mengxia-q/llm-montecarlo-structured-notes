"""Market inputs for the valuation template.

Everything here is held *fixed* between the gold-standard and the extracted
valuation of the same filing, so that the measured price deviation isolates
extraction error rather than market-data noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve, linearly interpolated in maturity.

    Built from the FRED constant-maturity Treasury series.  Rates are decimals
    (0.042 means 4.2%); flat extrapolation beyond the quoted tenors.
    """

    tenors: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.rates):
            raise ValueError("tenors and rates must have equal length")
        if len(self.tenors) == 0:
            raise ValueError("curve must contain at least one point")
        if list(self.tenors) != sorted(self.tenors):
            raise ValueError("tenors must be in increasing order")

    def zero(self, t: float | np.ndarray) -> float | np.ndarray:
        """Zero rate for maturity ``t`` (years)."""
        return np.interp(t, np.asarray(self.tenors), np.asarray(self.rates))

    def discount(self, t: float | np.ndarray) -> float | np.ndarray:
        """Discount factor to time ``t`` (years)."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)

    def forward_rates(self, times: np.ndarray) -> np.ndarray:
        """Piecewise-constant forward rates over ``[0, t_1], (t_1, t_2], ...``.

        Derived from the zero curve so that simulated growth is consistent with
        the discount factors used for the cash flows.
        """
        times = np.asarray(times, dtype=float)
        prev_t = np.concatenate([[0.0], times[:-1]])
        dt = times - prev_t
        if np.any(dt <= 0):
            raise ValueError("observation times must be strictly increasing and positive")
        integ = self.zero(times) * times
        prev_integ = np.concatenate([[0.0], integ[:-1]])
        return (integ - prev_integ) / dt

    @classmethod
    def flat(cls, rate: float) -> "ZeroCurve":
        """A flat curve, used by the unit tests and as a fallback."""
        return cls(tenors=(0.0, 100.0), rates=(rate, rate))


@dataclass(frozen=True)
class MarketInputs:
    """Per-filing market state as of the pricing date.

    Attributes
    ----------
    tickers:
        Market-data tickers in the same order as ``NoteTerms.underlyings``.
    spots:
        Closing level of each underlying on the pricing date, taken from market
        data and **not** from the filing.  This is what makes ``initial_level``
        a price-relevant field: performance is simulated as
        ``S(t) / initial_level`` with ``S(0) = spot``, so a mis-extracted
        initial level shifts every performance ratio and therefore the price.
    vols:
        Annualised volatilities estimated from the 252 trading days of daily log
        returns ending on the pricing date.
    corr:
        Correlation matrix of the same daily log returns, ordered like ``tickers``.
    div_yields:
        Continuous dividend yields, held constant per underlying.
    curve:
        Risk-free discount and growth curve.
    """

    tickers: tuple[str, ...]
    spots: np.ndarray
    vols: np.ndarray
    corr: np.ndarray
    div_yields: np.ndarray
    curve: ZeroCurve
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.tickers)
        for name, arr in (("spots", self.spots), ("vols", self.vols), ("div_yields", self.div_yields)):
            if np.shape(arr) != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {np.shape(arr)}")
        if np.shape(self.corr) != (n, n):
            raise ValueError(f"corr must have shape ({n}, {n}), got {np.shape(self.corr)}")
        if np.any(np.asarray(self.spots) <= 0):
            raise ValueError("spots must be positive")
        if np.any(np.asarray(self.vols) < 0):
            raise ValueError("vols must be non-negative")

    def cholesky(self) -> np.ndarray:
        """Lower-triangular Cholesky factor of the correlation matrix.

        Sample correlation matrices from unequal-length histories can be very
        slightly non-PSD; eigenvalues are floored before factorising and the
        matrix is renormalised to unit diagonal.
        """
        corr = np.array(self.corr, dtype=float)
        corr = 0.5 * (corr + corr.T)
        try:
            return np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.clip(eigvals, 1e-10, None)
            repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d = np.sqrt(np.diag(repaired))
            repaired = repaired / np.outer(d, d)
            return np.linalg.cholesky(repaired)
