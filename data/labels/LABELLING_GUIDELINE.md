# Labelling guideline v1

**Frozen:** 2026-10-06 (rule content); compiled into this file on 2026-10-10.
**Sources:** LABELLING RULES v1 in `experiments/prelabel_prompt.md`, the instructions given to the verifiers, and the extraction conventions in `src/extract/prompt.py`.
**Applies to:** all 50 labels under `data/labels/` (23 gold, 27 silver).

This guideline belongs to the current schema (`src/schema.py`, `src/extract/llm_schema.py`). If the schema changes, this guideline and every label must be re-checked.

---

## 1. Label types and how they are produced

| Type | Directory | Produced by |
|---|---|---|
| Prelabel | `prelabel/` | Claude (a Claude Code sub-agent, `prelabel:claude-opus-5-5`), one filing at a time, following sections 3–5. Every prelabel must pass `experiments/06_check_prelabels.py` |
| Gold | `gold/` | A human checks the prelabel field by field against the filing; `experiments/14_make_gold.py` then writes the gold label. The prelabel itself is never modified, so gold and silver can be compared |
| Silver | files in `prelabel/` with no gold counterpart | Not human-verified; reported separately in the evaluation |

The evaluation (`experiments/09_evaluate.py`) uses the gold label for a filing when there is one and the silver label otherwise. Every metric is reported both for all filings and for the gold-only subset.

Independence of prelabels: while labelling a filing, the annotator may read only that filing's cleaned text, the schema and the prompt code. It **must not open** anything else under `experiments/results/` or `data/labels/`, nor any other filing.

## 2. Scope

Products the pricing template can handle:
- contingent-coupon autocallable notes, with or without a memory coupon;
- the call is checked on discrete observation dates;
- downside protection observed only at maturity, or on every observation date;
- the underlying is a single equity **index or ETF**, or a worst-of basket of them, and each can be matched to market data in `src/market/tickers.py`.

The following are excluded, recorded in `exclusions.json` with the reason:
- buffered downside protection (losses only beyond a buffer), which the template `1 − g·(1 − P)` cannot express;
- daily knock-in monitoring;
- any underlying that is a single stock, a non-equity index, a risk-control index or an excess-return index;
- payoffs that are not contingent-coupon autocallables (for example participation-only or digital payoffs).

## 3. Field rules

All percentages are relative to each underlying's initial level and are written as the percentage itself: "70% of the Initial Level" is `70`, not `0.70`. Dates are `YYYY-MM-DD`. A value that the filing genuinely does not state is `null`: do not guess, interpolate or borrow from another field.

| Field | Rule |
|---|---|
| `issuer` | As written |
| `notional_per_unit` | The face amount per unit as written (10, 100 and 1000 all occur); **do not normalise** |
| `pricing_date` | Pricing date / trade date |
| `strike_date` | Only when the filing states a separate strike date / initial valuation date **that differs from the pricing date**; otherwise `null` |
| `underlyings[].name` | Copied verbatim, including symbols such as ®; spacing around ® does not matter |
| `underlyings[].initial_level` | The value as printed, **without rounding**, whichever date it is set on |
| `observation_dates` | **Every** coupon observation / valuation date, in order. Check the table row by row: do not skip or merge rows. If the filing says "expected" or "subject to postponement", use the scheduled date as printed |
| `payment_dates` | Same length as `observation_dates`, one to one. If the filing only gives a rule (for example "the third business day after each valuation date"), the dates **must be derived by program**: `~/.venvs/snllm/bin/python -m src.market.calendar <observation dates...> --plus N` (NYSE trading calendar), never by hand. Quote the rule as evidence, write "derived" in the note and add a flag |
| `autocall.first_observation_index` | The first observation date on which the note can be called, **counted from 1**. Many notes cannot be called in their first 6 or 12 months, so this is often greater than 1 |
| `autocall.levels_pct` | One value per callable observation date, in order. Repeat a constant level; write out a step-down ladder in full. **Whether the final observation date is callable follows the filing's wording exactly**: "interim valuation dates" excludes the final date, so the list is one shorter than the observation dates from `first_observation_index` on; "any Observation Date" includes it |
| `coupon.rate_per_period_pct` | The coupon actually paid **per period**, as a percentage of the face amount. A per-annum rate is divided by the number of payments per year; a cash amount per unit is converted to a percentage of `notional_per_unit`. Show the conversion in the evidence note |
| `coupon.barrier_pct` | Coupon barrier |
| `coupon.memory` | `true` only if unpaid coupons accumulate and are paid later when the barrier is met again. Keywords: "memory", "snowball", "previously unpaid", "all Contingent Coupon Payments not previously paid". `false` when each period stands alone |
| `knock_in.barrier_pct` | Principal protection level at maturity, often called Downside Threshold, Trigger Level or Barrier Level |
| `knock_in.monitoring` | `at_maturity` if observed only on the final valuation date; `observation_dates` if observed on every observation date; `continuous_daily` if observed every trading day (out of scope, see section 2) |
| `downside_participation_pct` | Participation in the decline below the protection level, as a percentage; the usual one-for-one loss is `100` |
| `estimated_value_per_unit` | The issuer's own estimated value per unit on the pricing date; `null` if only a range is given |
| `issue_date`, `maturity_date` | Issue date and maturity date |

**Critical fields** (checked one by one in evaluation and verification): `underlyings[].initial_level`, `observation_dates`, `autocall.first_observation_index`, `autocall.levels_pct`, `coupon.rate_per_period_pct`, `coupon.barrier_pct`, `coupon.memory`, `knock_in.barrier_pct`, `maturity_date` (`CRITICAL_FIELDS` in `src/schema.py`). Human verification also checks `payment_dates`, `knock_in.monitoring`, `pricing_date` and `strike_date`.

## 4. Evidence and flags (required in every prelabel)

- `evidence` has one entry per field (for `underlyings`, a `name` and an `initial_level` entry per underlying). Each entry has:
  - `quote`: a passage **copied verbatim** from the cleaned text, at most 300 characters; a script checks that it really occurs in the filing;
  - `lines`: the first and last line numbers in the cleaned text, counted from 1;
  - `note` (optional): any conversion made, for example `9.00% p.a. / 4 = 2.25`.
- When `strike_date` is `null`, quote the pricing date / trade date line as evidence.
- For `observation_dates`, quote the table header or first row, with `lines` covering the whole table.
- `flags`: add one for anything ambiguous, converted, split across pages, or stated inconsistently in the filing, naming the field and the reason. Verifiers rely on flags; when in doubt, add one.
- `scope`: whether the filing is in scope, and why.

## 5. Human verification (gold)

1. **Write the start time (HH:MM) in the timing sheet first**, then open the file. Verify one file at a time; do not alternate between files.
2. Open `terms` in the JSON and check every field against the filing. `source_file` is the cleaned text (tables rendered as Markdown); `raw_file` is the original HTML, whose tables are easier to read in a browser.
3. `evidence` only points to the right place; it **does not replace checking the value yourself**. Pay particular attention to `flags`.
4. **Do not assume the prelabel is right**: a file may contain a planted error.
5. Fix any error **directly in the `terms` values of the JSON**; no separate record is needed, because a script diffs the result.
6. **Write the end time as soon as you finish.** If you were away in between, note how many minutes in the comment column. Record times as you go; never estimate afterwards.
7. **Write comments in the timing sheet in English only.**
8. Files verified by both verifiers are done independently, without discussion.
9. While verifying, do not open: the seeded-error answer key, the other verifier's files, the untouched originals of your own files, `prelabel/`, `gold/` or `experiments/results/`.

After verification, write the gold label:
```bash
~/.venvs/snllm/bin/python experiments/14_make_gold.py <accession> --minutes <minutes> [--set field.path=JSON ...]
```
The second round was ingested as a batch by a script that also scored the seeded errors and the overlap files; its report is `data/labels/round2/round2_report.json`. The `provenance` of a gold label records the verifier, date, minutes, which fields were changed, whether the file was also verified by someone else, and whether an error was planted and found.

## 6. Comparison conventions in evaluation

These are not labelling rules, but they decide how labels are compared with model output, so they are stated here to avoid misreading:
- Fields ending in `_pct` have a tolerance of 0.001 percentage points (`PCT_TOLERANCE`), because some filings give both a coupon rate and a rounded cash amount, and either reading counts as correct.
- A sequence field is correct only if the whole sequence matches; date sequences also report element-level F1.
- "Price-equivalent": two call ladders that differ only in whether the final observation date is callable, with a last call level not below the knock-in barrier, count as equivalent, because the two readings price identically. The label itself still follows the filing's wording exactly, as in section 3.
