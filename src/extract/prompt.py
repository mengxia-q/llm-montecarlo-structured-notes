"""Prompt construction for term extraction.

The prompt carries the conventions that the schema's field descriptions cannot:
unit normalisation, what to do with per-annum versus per-period coupon rates, how
to expand a step-down call ladder, and the instruction to return null rather than
guess.  Field semantics themselves live in the ``Field(description=...)`` text of
``src.extract.llm_schema``, which is what the provider turns into a response
schema, so the two never drift apart.
"""

from __future__ import annotations

INSTRUCTIONS = """\
You are extracting the economic terms of a structured note from the final \
pricing supplement (SEC form 424B2) given below. Your output drives a \
Monte Carlo valuation model, so every number must be exactly what the document \
states.

Conventions you must follow:

1. Percentages are relative to each underlying's initial level, expressed as a \
percent number: write 70 for "70% of the Initial Level", not 0.70.
2. `coupon.rate_per_period_pct` is the coupon ACTUALLY PAID ON ONE PAYMENT DATE, \
in percent of principal. Filings usually quote a per-annum rate; divide it by the \
number of payments per year. If the filing quotes a cash amount per unit, convert \
it to a percent of `notional_per_unit`. State the per-period figure, never the \
per-annum one.
3. `observation_dates` must list EVERY coupon observation / valuation date in \
chronological order, and `payment_dates` must have the same length, each entry \
being the payment date for the observation date at the same position. These \
schedules are almost always given as a table; read the table row by row and do \
not skip rows or merge them.
4. `autocall.first_observation_index` is the 1-based position in \
`observation_dates` of the first date on which the note can be called. Many notes \
are not callable for the first six or twelve months, so this is often greater \
than 1.
5. `autocall.levels_pct` has exactly one entry per callable observation date, in \
chronological order. If the call level is constant, repeat it for every callable \
date. If it steps down, write out the full ladder. Note that many notes are \
callable only on the INTERIM valuation dates and not on the final valuation date \
("if, on any of the interim valuation dates, ..."); in that case this list has \
one entry fewer than the number of observation dates from \
`autocall.first_observation_index` onwards, and the final date is governed by the \
maturity rules instead.
6. `coupon.memory` is true ONLY if unpaid coupons accumulate and are paid on a \
later date when the barrier is met again. Wording to look for: "memory", \
"snowball", "previously unpaid", "all Contingent Coupon Payments not previously \
paid". If each period stands alone, it is false.
7. `knock_in.monitoring` is "at_maturity" when the downside barrier is tested only \
on the final valuation date, "observation_dates" when it is tested on each \
scheduled observation date, and "continuous_daily" when it is tested on every \
trading day during the life of the note.
8. `notional_per_unit` is the principal amount per unit as filed: issuers use 10, \
100 and 1000. Do not normalise it.
9. Dates are YYYY-MM-DD. `pricing_date` is the pricing / trade date. Some notes \
fix the initial levels on an earlier "strike date"; put that date in \
`strike_date`, and leave `strike_date` null if no separate strike date is named. \
`initial_level` is always the level as printed, whichever date it was set on.
10. If a value is genuinely absent from the document, return null for that field. \
Do not infer, interpolate, or carry a value over from a different field. An \
honest null is more useful than a plausible guess.

Return only the JSON object described by the response schema.
"""

RETRY_PREAMBLE = """\
A previous extraction of the document above failed deterministic validation \
checks. The failures are listed below. Re-read the relevant parts of the document \
and return a corrected extraction. Do not change fields that were not implicated.

Validation failures:
{failures}
"""


def build_prompt(document_text: str, *, failures: list[str] | None = None) -> str:
    """Assemble the extraction prompt.

    ``failures`` turns the prompt into the validator-guided retry (paper: Business Rules),
    where hard rule violations are fed back in natural language for one further
    attempt.  The feedback goes *after* the filing so that the retry shares the
    first attempt's byte-identical prefix and is served from the prompt cache.
    """
    parts = [INSTRUCTIONS, "=== BEGIN FILING ===", document_text, "=== END FILING ==="]
    if failures:
        bullets = "\n".join(f"- {failure}" for failure in failures)
        parts.append(RETRY_PREAMBLE.format(failures=bullets))
    return "\n\n".join(parts)
