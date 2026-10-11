# Pre-annotation sub-agent prompt

Used verbatim for the 2026-10-06 pilot (3 filings, all passed
`06_check_prelabels.py`). One sub-agent per filing. Fill in `{ACCESSION}`,
`{CLEANED_FILE}` (file name under `data/cleaned/`) and `{DESCRIPTION}`
(issuer, filing date, approximate tokens, underlyings from screening).

Since the pilot the schema gained `strike_date`; it is included below. The
prompt gave the repository's absolute local path, shown here as
`<repository root>`.

---

You are producing a PRE-ANNOTATION (a draft gold label that a human will verify field by field) for one SEC 424B2 structured-note pricing supplement, for a research paper on LLM term extraction + Monte Carlo pricing.

Repo root: <repository root>
Python: ~/.venvs/snllm/bin/python (run from repo root).

INPUT (the only filing you read): data/cleaned/{CLEANED_FILE}  ({DESCRIPTION}; tables are Markdown).
OUTPUT (the only file you write): data/labels/prelabel/{ACCESSION}.json

First read src/extract/prompt.py (INSTRUCTIONS = the extraction conventions you MUST follow: per-period coupon not per-annum, percent units relative to initial level, full observation schedule row by row, autocall.first_observation_index 1-based, levels_pct one entry per callable date and NOT including the final date if the note is callable only on interim dates, memory rule, monitoring rule, notional as filed, strike_date only if a separate strike date is named, null if absent) and src/extract/llm_schema.py (RawNoteTerms = the field set and descriptions). Read the whole filing carefully, every schedule table row.

INDEPENDENCE: do NOT open anything under experiments/results/, data/labels/ (other than writing your output), or any other filing. Other model outputs exist there and must not influence you.

Output JSON shape:
{
  "accession": "{ACCESSION}",
  "source_file": "data/cleaned/{CLEANED_FILE}",
  "annotator": "prelabel:claude-opus-5-5",
  "created": "<YYYY-MM-DD>",
  "terms": { ...exactly the RawNoteTerms fields: issuer, notional_per_unit, currency, pricing_date, strike_date, issue_date, maturity_date, underlyings[{name,ticker,initial_level}], performance_type, observation_dates[], payment_dates[], autocall{first_observation_index, levels_pct[]}, coupon{rate_per_period_pct, barrier_pct, memory}, knock_in{barrier_pct, monitoring}, downside_participation_pct, estimated_value_per_unit },
  "evidence": {
     "<field path>": {"quote": "<VERBATIM substring copied from the cleaned file, <=300 chars, enough to justify the value>", "lines": [<first line>, <last line>], "note": "<optional: any conversion you did, e.g. 9.00% p.a. / 4 = 2.25>"}
  },
  "flags": [ {"field": "<path>", "issue": "<why a human should look closely: ambiguity, conversion, table split, conflicting statements>"} ],
  "scope": {"in_scope": true|false, "reason": "<pricing template covers: European or observation-date knock-in, discrete autocall, memory/non-memory contingent coupon, single or worst-of on equity INDEX/ETF underlyings. Out of scope: daily knock-in monitoring, single-stock underlyings, participation/digital-only notes, total-return/excess-return/risk-control indices, etc.>"}
}
Evidence keys required (one each): issuer, notional_per_unit, pricing_date, strike_date (if null, quote the pricing/trade date line), issue_date, maturity_date, underlyings[i].name and underlyings[i].initial_level for each i, performance_type, observation_dates (quote a header/first row; "lines" = whole table span), payment_dates, autocall.first_observation_index, autocall.levels_pct, coupon.rate_per_period_pct, coupon.barrier_pct, coupon.memory, knock_in.barrier_pct, knock_in.monitoring, downside_participation_pct, estimated_value_per_unit. Quotes must be exact copies (they are programmatically checked); line numbers are 1-based lines of the cleaned file (use grep -n / Read line numbers). Use flags generously for anything uncertain — the human verifier relies on them. Do not invent values; null if genuinely absent.

LABELLING RULES (annotation guideline v1, 2026-10-06 — apply exactly):
- Payment dates given only as a rule (e.g. "the third business day after each valuation date"): derive them with `~/.venvs/snllm/bin/python -m src.market.calendar <obs dates...> --plus N` (NYSE calendar), never by hand; quote the rule as evidence, note "derived", and add a flag.
- Final valuation date callable or not: follow the filing's wording literally (interim-only => levels_pct excludes the final date).
- strike_date: only if the filing names a separate strike/initial-valuation date different from the pricing/trade date; else null.
- initial_level: exactly as printed (no rounding). Underlying names: verbatim incl. ® etc.
- coupon.rate_per_period_pct: always per period; convert per-annum or cash-amount coupons and show the arithmetic in the note.
- If dates in a schedule are "expected" or subject to postponement, use the scheduled dates as printed.
- Scope: out of scope if downside is buffered (loss only beyond a buffer), knock-in is monitored daily, any underlying is a single stock or a non-equity / risk-control / excess-return index, or the payoff is not a contingent-coupon autocallable.

Before finishing, run:
~/.venvs/snllm/bin/python experiments/06_check_prelabels.py data/labels/prelabel/{ACCESSION}.json
Fix and re-check if it fails (unless the failure reflects a true property of the filing — then flag it).

Final reply (short): the key extracted values (underlyings + initial levels, strike date, n observation dates, first callable index, #levels and their values, coupon per period, coupon barrier, memory, knock-in barrier + monitoring, maturity), scope verdict, and your list of flags.
