"""Screening out preliminary term sheets.

Inclusion criterion 5.1 of the method requires *final* pricing supplements: a
preliminary term sheet leaves the economic terms as ranges and the dates blank,
so there is no ground truth to extract.

File naming is not a reliable test - one sampled issuer names preliminary
documents ``*_prelim.htm`` but others do not - so the decision is made on the
document text.  Each signal is reported so that the exclusion can be audited and
the count reported in the paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..market.tickers import scope_name_patterns

#: Explicit self-identification.  Any one of these is conclusive.
_EXPLICIT = (
    "subject to completion",
    "preliminary term sheet",
    "preliminary pricing supplement",
    "preliminary prospectus supplement",
    "not complete and may be changed",
)

#: A bracketed range in place of a fixed term, e.g. "[$0.2375 to $0.25]" or
#: "[9.50% and 10.00%]".
_BRACKETED_RANGE = re.compile(
    r"\[[^\[\]]{0,60}?(?:\d[\d,.]*)\s*%?\s*(?:to|and|-|–)\s*(?:\$?\d[\d,.]*)[^\[\]]{0,20}?\]"
)

#: A month name followed by a blank day, e.g. "September , 2026".
_BLANK_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+,\s+\d{4}\b"
)

#: A currency amount left blank, e.g. "$ per $10 principal amount".
_BLANK_AMOUNT = re.compile(r"\$\s+per\s+\$?\d")

#: An "expected to be between X and Y" estimated value, which only preliminary
#: documents carry; the final supplement states a single number.
_ESTIMATED_RANGE = re.compile(
    r"estimated value[^.]{0,120}?expected to be between", re.IGNORECASE
)


@dataclass(frozen=True)
class PreliminaryVerdict:
    is_preliminary: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.is_preliminary


def is_preliminary(text: str, *, min_soft_signals: int = 2) -> PreliminaryVerdict:
    """Decide whether ``text`` is a preliminary term sheet.

    An explicit self-identification decides on its own.  Otherwise at least
    ``min_soft_signals`` structural signals (bracketed ranges, blank dates, blank
    amounts, a ranged estimated value) are required, so that a single stray
    bracket does not discard a usable final supplement.
    """
    low = text.lower()
    reasons: list[str] = []

    for phrase in _EXPLICIT:
        if phrase in low:
            return PreliminaryVerdict(True, [f"explicit phrase: {phrase!r}"])

    soft: list[str] = []
    n_ranges = len(_BRACKETED_RANGE.findall(text))
    if n_ranges:
        soft.append(f"{n_ranges} bracketed term range(s)")
    n_blank_dates = len(_BLANK_DATE.findall(text))
    if n_blank_dates:
        soft.append(f"{n_blank_dates} blank date(s)")
    if _BLANK_AMOUNT.search(text):
        soft.append("blank per-unit amount")
    if _ESTIMATED_RANGE.search(text):
        soft.append("estimated value given as a range")

    if len(soft) >= min_soft_signals:
        return PreliminaryVerdict(True, soft)
    return PreliminaryVerdict(False, soft)



#: How much of the document to inspect for the underlying description.  The
#: "Linked to the ..." title and the terms table both sit near the front; reading
#: further would pick up the index descriptions in the risk-factor discussion of
#: notes that are not actually linked to an index.
SCOPE_HEAD_CHARS = 6_000

#: Phrases that mark a single-stock underlying, which is out of scope: a
#: constant-dividend-yield GBM template cannot handle discrete dividends and
#: split adjustment (paper: Pricing Template).
_SINGLE_STOCK_MARKERS = (
    "common stock of",
    "shares of common stock",
    "class a common stock",
    "ordinary shares",
    "american depositary shares",
)


@dataclass(frozen=True)
class ScopeVerdict:
    in_scope: bool
    matched_underlyings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.in_scope


def in_underlying_scope(text: str, *, head_chars: int = SCOPE_HEAD_CHARS) -> ScopeVerdict:
    """Decide whether a filing's underlyings are inside the dataset scope.

    The dataset is restricted to equity index and ETF underlyings (see
    ``src.market.tickers``).  This screen runs on the cleaned text *before* any
    extraction call, so filings linked to single stocks cost nothing.  It is a
    pre-filter, not the authority: ``resolve_underlying`` makes the final
    decision once the underlyings have actually been extracted.
    """
    head = text[:head_chars].lower()
    reasons: list[str] = []

    matched = [pattern for pattern in scope_name_patterns() if pattern in head]
    if not matched:
        reasons.append("no registered index or ETF name in the document head")

    stock_markers = [marker for marker in _SINGLE_STOCK_MARKERS if marker in head]
    if stock_markers:
        reasons.append(f"single-stock wording: {stock_markers[0]!r}")

    return ScopeVerdict(
        in_scope=bool(matched) and not stock_markers,
        matched_underlyings=matched,
        reasons=reasons,
    )


#: Worst-of wording.  Any one marks a basket whose performance is the minimum.
_WORST_OF_MARKERS = (
    "least performing",
    "least-performing",
    "worst performing",
    "worst-performing",
    "lesser performing",
    "lesser-performing",
    "lowest performing",
)

#: Memory-coupon wording.  "Unpaid contingent coupon" alone is not enough: the
#: non-memory notes use it too ("you will not receive the unpaid ...").
_MEMORY_MARKERS = ("previously unpaid", "memory")

#: Downside shapes the template does not price: a buffered loss shifts the
#: final payoff by a buffer amount, which ``1 - g * (1 - P)`` cannot express.
_BUFFER_MARKERS = ("buffer amount", "buffer rate", "buffer percentage")

#: Barrier monitoring on every trading day rather than on observation dates.
_DAILY_MONITORING = re.compile(
    r"(?:on any|on each|any) (?:scheduled )?trading day (?:during|from|in) the "
    r"(?:term|monitoring period|observation period)[^.]{0,200}?"
    r"(?:barrier|trigger|threshold|knock)",
    re.IGNORECASE,
)

_ROW_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}\b"
)
_ROW_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_STEP_DOWN_WORDING = re.compile(
    r"step[- ]down|(?:call|autocall|redemption) (?:threshold|level|barrier|trigger)s?"
    r"[^.]{0,80}?(?:declin|decreas|step)",
    re.IGNORECASE,
)


def _has_step_down_schedule(text: str) -> bool:
    """True if some dated table column holds a percentage that falls over time.

    A step-down autocall prints its call levels in the observation schedule
    table, one per dated row.  A column whose values are non-increasing with at
    least two distinct levels across three or more rows is taken as a ladder.
    This is a sampling heuristic for stratification only; the extraction and the
    labels decide the actual levels.
    """
    columns: dict[int, list[float]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or not _ROW_DATE.search(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for idx, cell in enumerate(cells):
            m = _ROW_PCT.search(cell) if "%" in cell else None
            if m and len(cell) < 40:
                columns.setdefault(idx, []).append(float(m.group(1)))
    for values in columns.values():
        if (
            len(values) >= 3
            and len(set(values)) >= 2
            and all(a >= b for a, b in zip(values, values[1:]))
        ):
            return True
    return False


#: Index families named in a title that the ticker registry cannot price.  The
#: underlying-scope screen only needs *one* registered name, so a worst-of on
#: the S&P 500 and the MSCI Emerging Markets Index passes it; this catches the
#: unregistered leg.  "iShares MSCI ..." ETFs are registered and excluded here.
_UNREGISTERED_INDEX = re.compile(
    r"(?<!ishares )\bmsci\b[^.,;]{0,40}?index|\bstoxx\b(?! 50)|"
    r"\b(?:kospi|ibovespa|cac 40|dax|asx 200|topix)\b"
)


def _unregistered_index(head: str) -> str | None:
    m = _UNREGISTERED_INDEX.search(head)
    return m.group(0) if m else None


@dataclass(frozen=True)
class ProductProfile:
    """Coarse product features used to stratify the sample (paper: Dataset)."""

    worst_of: bool
    memory: bool
    step_down: bool
    buffered: bool
    daily_monitoring: bool
    unregistered_index: str | None = None

    def to_dict(self) -> dict:
        return {
            "worst_of": self.worst_of,
            "memory": self.memory,
            "step_down": self.step_down,
            "buffered": self.buffered,
            "daily_monitoring": self.daily_monitoring,
            "unregistered_index": self.unregistered_index,
        }


def product_profile(text: str, *, matched_underlyings: list[str] | None = None) -> ProductProfile:
    """Heuristic product features read from the cleaned text.

    These tags drive stratified sampling and the out-of-template exclusion
    count.  They are not labels: memory and step-down are read again from the
    filing during annotation.
    """
    low = text.lower()
    head = low[:SCOPE_HEAD_CHARS * 3]
    worst_of = any(m in low for m in _WORST_OF_MARKERS) or len(matched_underlyings or []) > 1
    return ProductProfile(
        worst_of=worst_of,
        memory=any(m in head for m in _MEMORY_MARKERS),
        step_down=bool(_STEP_DOWN_WORDING.search(head)) or _has_step_down_schedule(text),
        buffered=any(m in head for m in _BUFFER_MARKERS),
        daily_monitoring=bool(_DAILY_MONITORING.search(text)),
        unregistered_index=_unregistered_index(low[:SCOPE_HEAD_CHARS]),
    )
