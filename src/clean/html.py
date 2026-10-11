"""424B2 HTML -> high signal-to-noise plain text, with tables kept as Markdown.

Three things matter for extraction quality, in order:

1. **Tables must survive.**  Observation-date schedules and step-down call levels
   live almost exclusively in tables, and ``BeautifulSoup.get_text()`` flattens a
   table into an unordered run of numbers that no model can realign.  Each
   ``<table>`` is therefore rendered as a Markdown pipe table, preserving row and
   column structure.

2. **Running headers and footers must go.**  A term sheet repeats its product
   title on every page; the raw text of one sampled GS filing contained the same
   heading 11 times.  Lines that repeat verbatim many times are dropped.

3. **Boilerplate sections may go, by heading, but never at the cost of a term.**
   Risk factors, tax discussion, ERISA, supplemental plan of distribution and the
   validity-of-notes opinion carry no pricing parameters.  Dropping them is
   nevertheless dangerous: section boundaries in these filings are not reliably
   marked, and on one sampled Bank of Montreal filing a naive
   heading-to-next-heading rule swallowed the terms as well, cutting
   "Coupon Barrier" from 18 occurrences to 8 and "Pricing Date" from 8 to 2.
   ``drop_boilerplate`` therefore *refuses* to remove any block that still
   contains pricing-term vocabulary, and ``clean_filing`` reports both the
   sections it dropped and the ones it refused to drop, so the paper's Cleaning
   section can state the rule and the before/after token counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

#: Section headings whose content carries no pricing parameters.  Matched
#: case-insensitively against the start of a heading-like line.
BOILERPLATE_HEADINGS: tuple[str, ...] = (
    "risk factors",
    "additional risk factors",
    "selected risk considerations",
    "united states federal income tax",
    "u.s. federal income tax",
    "material u.s. federal income tax",
    "supplemental discussion of federal income tax",
    "certain employee benefit plan considerations",
    "employee retirement income security act",
    "erisa",
    "supplemental plan of distribution",
    "validity of the notes",
    "benefit plan investor considerations",
    "where you can find more information",
    "incorporation of information by reference",
)

#: A line repeated at least this many times is treated as a running header.
RUNNING_HEADER_MIN_REPEATS = 4

#: Vocabulary that marks text as carrying pricing parameters.  A candidate
#: boilerplate block containing any of these is kept: losing a term is far more
#: damaging to the experiment than carrying a few thousand extra tokens.
TERM_VOCABULARY: tuple[str, ...] = (
    "contingent coupon",
    "coupon barrier",
    "coupon payment date",
    "observation date",
    "valuation date",
    "call level",
    "call value",
    "autocall",
    "automatic call",
    "initial level",
    "initial underlier level",
    "starting value",
    "trigger level",
    "threshold value",
    "knock-in",
    "downside threshold",
    "pricing date",
    "maturity date",
    "principal amount per",
    "estimated value",
)

_TOKEN_DIVISOR = 4  # rough chars-per-token for reporting only


@dataclass(frozen=True)
class DroppedSection:
    heading: str
    chars_removed: int


@dataclass(frozen=True)
class RefusedSection:
    """A boilerplate section that was kept because it still contained terms."""

    heading: str
    chars_kept: int
    matched_terms: tuple[str, ...]


@dataclass
class CleanedDocument:
    text: str
    n_tables: int
    chars_raw: int
    chars_clean: int
    dropped_sections: list[DroppedSection] = field(default_factory=list)
    refused_sections: list[RefusedSection] = field(default_factory=list)
    running_headers_removed: int = 0

    @property
    def approx_tokens_raw(self) -> int:
        return self.chars_raw // _TOKEN_DIVISOR

    @property
    def approx_tokens_clean(self) -> int:
        return self.chars_clean // _TOKEN_DIVISOR

    @property
    def reduction_pct(self) -> float:
        if self.chars_raw == 0:
            return 0.0
        return 100.0 * (1.0 - self.chars_clean / self.chars_raw)


def _cell_text(cell: Tag) -> str:
    text = cell.get_text(" ", strip=True)
    text = text.replace("|", "\\|")
    return re.sub(r"\s+", " ", text)


def _span(cell: Tag, attr: str) -> int:
    value = cell.get(attr)
    try:
        return max(1, int(value)) if value else 1
    except (TypeError, ValueError):
        return 1


def table_to_markdown(table: Tag) -> str:
    """Render one ``<table>`` as a Markdown pipe table.

    Both ``colspan`` and ``rowspan`` are expanded by repeating the cell text, so
    every value stays under its own header.  Tracking ``rowspan`` is not
    optional: 50 of 96 sampled filings use it, typically in the cover table of a
    worst-of note where the coupon and CUSIP cells span every underlying's row.
    Left untracked, the second underlying's row loses those columns and its
    initial level slides under the "Contingent Coupon" header (observed on GS and
    TD filings).  Fully empty rows and columns are dropped - filings use them as
    spacers and they otherwise dominate the output.
    """
    rows: list[list[str]] = []
    # Column index -> (text, rows still to fill) for cells spanning down.
    carried: dict[int, tuple[str, int]] = {}
    for tr in table.find_all("tr"):
        row: dict[int, str] = {col: text for col, (text, _) in carried.items()}
        carried = {col: (text, left - 1) for col, (text, left) in carried.items() if left > 1}
        col = 0
        for cell in tr.find_all(["td", "th"]):
            text = _cell_text(cell)
            down = _span(cell, "rowspan")
            for _ in range(_span(cell, "colspan")):
                while col in row:
                    col += 1
                row[col] = text
                if down > 1:
                    carried[col] = (text, down - 1)
                col += 1
        rows.append([row.get(i, "") for i in range(max(row) + 1)] if row else [])

    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    keep = [i for i in range(width) if any(r[i] for r in rows)]
    if not keep:
        return ""
    rows = [[r[i] for i in keep] for r in rows]

    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def html_to_text(html: str) -> tuple[str, int]:
    """Plain text with every table replaced by its Markdown rendering.

    Returns the text and the number of tables converted.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    tables = soup.find_all("table")
    for table in tables:
        markdown = table_to_markdown(table)
        placeholder = f"\n\n{markdown}\n\n" if markdown else "\n"
        table.replace_with(NavigableString(placeholder))

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), len(tables)


def _strip_running_headers(text: str, min_repeats: int) -> tuple[str, int]:
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        # Only short, non-tabular, non-numeric lines can be running headers.
        if 8 <= len(stripped) <= 120 and "|" not in stripped and not stripped[0].isdigit():
            counts[stripped] = counts.get(stripped, 0) + 1

    repeated = {k for k, v in counts.items() if v >= min_repeats}
    if not repeated:
        return text, 0

    kept, removed = [], 0
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped in repeated:
            # Keep the first occurrence - it is usually the real product title.
            if stripped in seen:
                removed += 1
                continue
            seen.add(stripped)
        kept.append(line)
    return "\n".join(kept), removed


_HEADING_RE = re.compile(r"^[^a-z]*([A-Za-z][A-Za-z ,.'\-/&]{3,80})\s*$")


def _is_heading_for(line: str, headings: tuple[str, ...]) -> str | None:
    stripped = line.strip().strip("_*#").rstrip(":").strip()
    if not stripped or len(stripped) > 120 or "|" in stripped:
        return None
    low = re.sub(r"\s+", " ", stripped.lower())
    low = re.sub(r"^[0-9ivxlc.\s\-]+", "", low)
    for heading in headings:
        if low.startswith(heading):
            return heading
    return None


def _matched_terms(block: str) -> tuple[str, ...]:
    low = block.lower()
    return tuple(term for term in TERM_VOCABULARY if term in low)


def drop_boilerplate(
    text: str, headings: tuple[str, ...] = BOILERPLATE_HEADINGS
) -> tuple[str, list[DroppedSection], list[RefusedSection]]:
    """Remove boilerplate sections, from their heading to the next heading.

    A candidate block runs from a boilerplate heading to the next line that looks
    like a section heading.  That boundary detection is unreliable in these
    filings, so a block is dropped only if it contains no pricing-term
    vocabulary; otherwise it is kept and recorded as refused.
    """
    lines = text.split("\n")
    kept: list[str] = []
    dropped: list[DroppedSection] = []
    refused: list[RefusedSection] = []

    i = 0
    while i < len(lines):
        heading = _is_heading_for(lines[i], headings)
        if heading is None:
            kept.append(lines[i])
            i += 1
            continue

        start = i
        i += 1
        while i < len(lines):
            candidate = lines[i].strip().strip("_*#").rstrip(":").strip()
            if (
                candidate
                and "|" not in candidate
                and len(candidate) <= 80
                and _HEADING_RE.match(candidate)
                and _is_heading_for(lines[i], headings) is None
                and candidate[:1].isupper()
                and sum(c.islower() for c in candidate) < 0.7 * len(candidate)
            ):
                break
            i += 1

        block_lines = lines[start:i]
        block = "\n".join(block_lines)
        terms = _matched_terms(block)
        if terms:
            refused.append(
                RefusedSection(
                    heading=heading, chars_kept=len(block), matched_terms=terms
                )
            )
            kept.extend(block_lines)
            continue

        dropped.append(DroppedSection(heading=heading, chars_removed=len(block) + 1))

    return "\n".join(kept), dropped, refused


def clean_filing(
    html: str,
    *,
    drop_sections: bool = True,
    strip_running_headers: bool = True,
) -> CleanedDocument:
    """Full cleaning pipeline for one 424B2 filing.

    ``drop_sections`` and ``strip_running_headers`` are exposed as flags so that
    a cleaning ablation can quantify what each step removes.
    """
    text, n_tables = html_to_text(html)
    chars_raw = len(text)

    headers_removed = 0
    if strip_running_headers:
        text, headers_removed = _strip_running_headers(text, RUNNING_HEADER_MIN_REPEATS)

    dropped: list[DroppedSection] = []
    refused: list[RefusedSection] = []
    if drop_sections:
        text, dropped, refused = drop_boilerplate(text)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return CleanedDocument(
        text=text,
        n_tables=n_tables,
        chars_raw=chars_raw,
        chars_clean=len(text),
        dropped_sections=dropped,
        refused_sections=refused,
        running_headers_removed=headers_removed,
    )
