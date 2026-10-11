"""Cleaning: table structure survives, and boilerplate removal never eats a term."""

from src.clean import clean_filing, drop_boilerplate, html_to_text
from src.clean.html import table_to_markdown
from bs4 import BeautifulSoup


def test_tables_become_markdown_with_rows_and_columns_intact():
    html = """
    <table>
      <tr><th>Observation date</th><th>Call level</th></tr>
      <tr><td>June 16, 2026</td><td>100%</td></tr>
      <tr><td>September 15, 2026</td><td>95%</td></tr>
    </table>
    """
    text, n_tables = html_to_text(html)
    assert n_tables == 1
    assert "| Observation date | Call level |" in text
    assert "| June 16, 2026 | 100% |" in text
    assert "| September 15, 2026 | 95% |" in text


def test_spacer_rows_and_columns_are_dropped():
    html = """
    <table>
      <tr><td>Date</td><td></td><td>Level</td></tr>
      <tr><td></td><td></td><td></td></tr>
      <tr><td>2026-06-16</td><td></td><td>100%</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "lxml").find("table")
    markdown = table_to_markdown(table)
    assert markdown.splitlines()[0] == "| Date | Level |"
    assert len(markdown.splitlines()) == 3  # header, rule, one body row


def test_colspan_is_expanded_so_columns_stay_aligned():
    html = '<table><tr><td colspan="2">Schedule</td></tr><tr><td>A</td><td>B</td></tr></table>'
    table = BeautifulSoup(html, "lxml").find("table")
    assert table_to_markdown(table).splitlines()[0] == "| Schedule | Schedule |"


def test_rowspan_is_carried_down_so_values_stay_under_their_header():
    """Regression test for the GS / TD worst-of cover table.

    The coupon cell spans both underlyings' rows.  Untracked, the second row
    lost that column and its initial level slid under "Contingent Coupon".
    """
    html = """
    <table>
      <tr><th>Index</th><th>Contingent Coupon</th><th>Initial Level</th><th>CUSIP</th></tr>
      <tr><td>Russell 2000</td><td rowspan="2">$0.20075/qtr.</td><td>2,992.542</td>
          <td rowspan="2">36274H642</td></tr>
      <tr><td>S&amp;P 500</td><td>7,543.64</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "lxml").find("table")
    lines = table_to_markdown(table).splitlines()
    assert lines[2] == "| Russell 2000 | $0.20075/qtr. | 2,992.542 | 36274H642 |"
    assert lines[3] == "| S&P 500 | $0.20075/qtr. | 7,543.64 | 36274H642 |"


def test_rowspan_and_colspan_combine():
    html = (
        '<table><tr><td rowspan="2" colspan="2">Block</td><td>a</td></tr>'
        "<tr><td>b</td></tr><tr><td>x</td><td>y</td><td>z</td></tr></table>"
    )
    table = BeautifulSoup(html, "lxml").find("table")
    assert table_to_markdown(table).splitlines()[2:] == [
        "| Block | Block | b |",
        "| x | y | z |",
    ]


def test_pipes_inside_cells_are_escaped():
    html = "<table><tr><td>a|b</td><td>c</td></tr></table>"
    table = BeautifulSoup(html, "lxml").find("table")
    assert r"a\|b" in table_to_markdown(table)


def test_boilerplate_without_terms_is_dropped():
    text = "\n".join([
        "Terms Of The Notes",
        "Coupon Barrier: 70% of the Initial Level",
        "Risk Factors",
        "An investment in the notes involves significant risks.",
        "You may lose some or all of your investment.",
        "Supplemental Information",
        "Other matters.",
    ])
    cleaned, dropped, refused = drop_boilerplate(text)
    assert [d.heading for d in dropped] == ["risk factors"]
    assert refused == []
    assert "significant risks" not in cleaned
    assert "Coupon Barrier: 70%" in cleaned


def test_boilerplate_containing_terms_is_refused_not_dropped():
    """Regression test for the Bank of Montreal failure.

    Section boundaries are unreliable in these filings, so a block that still
    mentions pricing terms must be kept even though its heading is boilerplate.
    """
    text = "\n".join([
        "Selected Risk Considerations",
        "The notes may not pay a coupon.",
        "Coupon Barrier: 70% of the Initial Level",
        "Observation Date schedule follows.",
    ])
    cleaned, dropped, refused = drop_boilerplate(text)
    assert dropped == []
    assert [r.heading for r in refused] == ["selected risk considerations"]
    assert "coupon barrier" in [t for t in refused[0].matched_terms]
    assert "Coupon Barrier: 70%" in cleaned


def test_running_headers_are_removed_but_the_first_occurrence_is_kept():
    header = "Autocallable Contingent Coupon Barrier Notes"
    body = []
    for i in range(6):
        body += [header, f"page {i} content"]
    doc = clean_filing(f"<html><body><p>{'</p><p>'.join(body)}</p></body></html>")
    assert doc.text.count(header) == 1
    assert doc.running_headers_removed == 5


def test_reduction_is_reported_against_the_pre_cleaning_text():
    doc = clean_filing("<html><body><p>Coupon Barrier: 70%</p></body></html>")
    assert doc.chars_raw > 0
    assert doc.reduction_pct == 0.0
    assert doc.approx_tokens_clean == doc.chars_clean // 4
