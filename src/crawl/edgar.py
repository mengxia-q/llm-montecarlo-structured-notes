"""EDGAR full-text search and filing download.

SEC access rules the client follows: a descriptive ``User-Agent`` carrying a
contact address, and no more than ten requests per second (the client is
deliberately slower than that).  Responses are gzip-encoded; the full-text
search endpoint returns an error page rather than JSON when the request is
malformed, so every response is checked.

The full-text search index caps ``hits.total`` at 10,000 and returns at most 100
hits per page with a 10,000-hit deep-paging limit, so broad queries are sliced by
month to enumerate more than that.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

import requests

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

#: Seconds between requests.  The SEC limit is 10/s; this is far below it.
REQUEST_INTERVAL = 0.25

#: EDGAR full-text search refuses to deep-page past this offset.
MAX_FROM = 9_900

#: Frequent issuers of this product class, by CIK, for the stratified sampling
#: described in the paper's Dataset section.  Full-text search orders hits by relevance, which clusters results
#: on whichever issuers file most; querying each CIK separately is what produces
#: a balanced sample across issuers.
ISSUER_CIKS: dict[str, str] = {
    "UBS AG": "0001114446",
    "Barclays Bank PLC": "0000312070",
    "Citigroup Global Markets Holdings Inc.": "0000200245",
    "GS Finance Corp.": "0001419828",
    "HSBC USA Inc.": "0000083246",
    "Bank of Nova Scotia": "0000009631",
    "Bank of Montreal": "0000927971",
    "Toronto-Dominion Bank": "0000947263",
    "JPMorgan Chase Financial Co. LLC": "0001665650",
    "Morgan Stanley Finance LLC": "0000895421",
    "BofA Finance LLC": "0001682472",
    "Royal Bank of Canada": "0001000275",
}


@dataclass(frozen=True)
class FilingRef:
    """One filing returned by full-text search."""

    accession: str
    document: str
    cik: str
    entity: str
    filing_date: date
    form: str

    @property
    def filing_id(self) -> str:
        """Stable identifier used for cache filenames and dataset keys."""
        return f"{self.accession}_{self.document}"

    @property
    def is_pdf(self) -> bool:
        return self.document.lower().endswith(".pdf")

    def as_html(self) -> "FilingRef":
        """The same filing, pointing at its HTML rendition.

        About a quarter of the 424B2 hits in this product class are returned by
        full-text search as PDFs.  Every one inspected also ships an HTML
        document with the identical basename in the same accession folder, so the
        PDFs cost no sample: the extension is swapped and the result verified by
        ``EdgarClient.resolve_html``.
        """
        if not self.is_pdf:
            return self
        return replace(self, document=self.document[: -len(".pdf")] + ".htm")

    @property
    def url(self) -> str:
        return f"{ARCHIVES}/{self.cik.lstrip('0')}/{self.accession.replace('-', '')}/{self.document}"

    @property
    def index_url(self) -> str:
        """Human-readable filing index page, for citing the source in the paper."""
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik.lstrip('0')}/"
            f"{self.accession.replace('-', '')}/{self.accession}-index.htm"
        )

    def to_dict(self) -> dict:
        return {
            "accession": self.accession,
            "document": self.document,
            "cik": self.cik,
            "entity": self.entity,
            "filing_date": self.filing_date.isoformat(),
            "form": self.form,
            "url": self.url,
            "index_url": self.index_url,
        }


class EdgarError(RuntimeError):
    pass


class EdgarClient:
    """Thin, rate-limited EDGAR client.

    Parameters
    ----------
    contact:
        Contact address embedded in the ``User-Agent`` header, as the SEC
        requires.
    raw_dir:
        Directory for downloaded HTML.  Downloads are cached, so re-running an
        experiment does not re-hit EDGAR and the dataset stays reproducible.
    """

    def __init__(
        self,
        contact: str,
        *,
        raw_dir: Path = DEFAULT_RAW_DIR,
        interval: float = REQUEST_INTERVAL,
    ) -> None:
        if "@" not in contact:
            raise ValueError("contact must be an email address, as the SEC requires")
        self.raw_dir = raw_dir
        self.interval = interval
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"structured-notes-llm research {contact}",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    # -- plumbing ---------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, params: dict | None = None, attempts: int = 4):
        last: Exception | None = None
        for attempt in range(attempts):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last = exc
                if attempt < attempts - 1:
                    time.sleep(2.0 * (attempt + 1))
        raise EdgarError(f"GET {url} failed after {attempts} attempts: {last}") from last

    # -- search -----------------------------------------------------------
    def search_page(
        self,
        query: str,
        *,
        forms: str = "424B2",
        start: date | None = None,
        end: date | None = None,
        offset: int = 0,
        ciks: str | None = None,
    ) -> tuple[list[FilingRef], int]:
        """One page of full-text search results, plus the reported total."""
        params: dict[str, str | int] = {"q": query, "forms": forms, "from": offset}
        if ciks:
            params["ciks"] = ciks
        if start and end:
            params |= {
                "dateRange": "custom",
                "startdt": start.isoformat(),
                "enddt": end.isoformat(),
            }

        response = self._get(FTS_URL, params=params)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise EdgarError(
                f"full-text search returned non-JSON for q={query!r}: "
                f"{response.text[:200]!r}"
            ) from exc

        hits = payload.get("hits", {})
        total = hits.get("total", {}).get("value", 0)
        refs = []
        for hit in hits.get("hits", []):
            source = hit.get("_source", {})
            accession, _, document = hit.get("_id", "").partition(":")
            ciks = source.get("ciks") or []
            if not accession or not document or not ciks:
                continue
            names = source.get("display_names") or ["?"]
            refs.append(
                FilingRef(
                    accession=accession,
                    document=document,
                    cik=ciks[0],
                    entity=names[0],
                    filing_date=date.fromisoformat(source["file_date"]),
                    form=(source.get("file_type") or forms),
                )
            )
        return refs, total

    def search(
        self,
        query: str,
        *,
        forms: str = "424B2",
        start: date,
        end: date,
        limit: int = 200,
        month_slices: bool = True,
    ) -> list[FilingRef]:
        """Enumerate search results, optionally slicing the window by month.

        Month slicing exists because the index refuses to deep-page past 10,000
        hits; one month of 424B2 autocallable filings is comfortably below that.
        """
        windows: list[tuple[date, date]] = []
        if month_slices:
            cursor = start
            while cursor <= end:
                nxt = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
                windows.append((cursor, min(nxt - timedelta(days=1), end)))
                cursor = nxt
        else:
            windows.append((start, end))

        found: dict[str, FilingRef] = {}
        for win_start, win_end in windows:
            offset = 0
            while len(found) < limit:
                refs, total = self.search_page(
                    query, forms=forms, start=win_start, end=win_end, offset=offset
                )
                if not refs:
                    break
                for ref in refs:
                    found.setdefault(ref.filing_id, ref)
                    if len(found) >= limit:
                        break
                offset += len(refs)
                if offset >= min(total, MAX_FROM):
                    break
            if len(found) >= limit:
                break
        return list(found.values())

    def search_by_issuer(
        self,
        query: str,
        *,
        start: date,
        end: date,
        per_issuer: int,
        issuers: dict[str, str] | None = None,
        forms: str = "424B2",
    ) -> dict[str, list[FilingRef]]:
        """Run ``query`` once per issuer CIK, returning up to ``per_issuer`` each.

        This is the sampling entry point for the dataset: relevance-ordered
        full-text search clusters on whichever issuers file most, so a single
        query cannot produce the issuer stratification the dataset requires.
        Issuers with no matching filing in the window map to an empty list.
        """
        issuers = issuers or ISSUER_CIKS
        out: dict[str, list[FilingRef]] = {}
        for name, cik in issuers.items():
            refs: dict[str, FilingRef] = {}
            offset = 0
            while len(refs) < per_issuer:
                page, total = self.search_page(
                    query, forms=forms, start=start, end=end, offset=offset, ciks=cik
                )
                if not page:
                    break
                for ref in page:
                    refs.setdefault(ref.filing_id, ref)
                offset += len(page)
                if offset >= min(total, MAX_FROM):
                    break
            out[name] = list(refs.values())[:per_issuer]
        return out

    def filing_documents(self, ref: FilingRef) -> list[str]:
        """Every document name in the filing's accession folder."""
        url = (
            f"{ARCHIVES}/{ref.cik.lstrip('0')}/"
            f"{ref.accession.replace('-', '')}/index.json"
        )
        payload = self._get(url).json()
        return [item["name"] for item in payload["directory"]["item"]]

    def resolve_html(self, ref: FilingRef) -> FilingRef:
        """Return ``ref`` pointing at an HTML document, or raise.

        Tries the same-basename ``.htm`` sibling first, since that is how the
        filing agents name them, and falls back to scanning the accession folder
        for a ``*424b2*.htm`` document.
        """
        if not ref.is_pdf:
            return ref

        candidate = ref.as_html()
        try:
            self._get(candidate.url, attempts=1)
            return candidate
        except EdgarError:
            pass

        names = [n for n in self.filing_documents(ref) if n.lower().endswith((".htm", ".html"))]
        stem = ref.document.rsplit(".", 1)[0].lower()
        for name in names:
            if name.rsplit(".", 1)[0].lower() == stem:
                return replace(ref, document=name)
        for name in names:
            low = name.lower()
            if "424b2" in low and not low.startswith(ref.accession):
                return replace(ref, document=name)
        raise EdgarError(f"no HTML rendition found for {ref.accession}/{ref.document}")

    # -- download ---------------------------------------------------------
    def download(self, ref: FilingRef, *, refresh: bool = False) -> str:
        """Filing HTML, cached under ``raw_dir``.

        Raises if the document is not HTML: feeding PDF bytes to an HTML parser
        silently yields tens of thousands of tokens of PDF internals rather than
        an error, which would quietly corrupt the dataset.  Call
        ``resolve_html`` first.
        """
        if ref.is_pdf:
            raise EdgarError(
                f"{ref.document} is a PDF; call resolve_html(ref) to get the "
                "HTML rendition of this filing"
            )

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{ref.filing_id}"
        if path.exists() and not refresh:
            html = path.read_text(encoding="utf-8", errors="replace")
        else:
            html = self._get(ref.url).text
            path.write_text(html, encoding="utf-8")

        if html.lstrip()[:5].startswith("%PDF"):
            raise EdgarError(f"{ref.document} served PDF content despite its extension")
        return html
