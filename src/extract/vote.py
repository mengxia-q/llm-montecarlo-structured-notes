"""Field-level self-consistency voting and document confidence (paper: Voting and Confidence).

``k`` independent extractions are reduced field by field, taking the majority
value.  Sequence-valued fields vote as a whole - a date schedule is either right
or wrong, and a majority over individual positions would synthesise a schedule
that no sample actually proposed.

The agreement rate of a field is the majority's vote share.  A document's
confidence is the lowest agreement rate among the critical fields, multiplied by
a penalty if deterministic validation raised soft warnings.  Taking the minimum
rather than the mean is deliberate: one unreliable critical field is enough to
make the resulting valuation untrustworthy.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..schema import CRITICAL_FIELDS

#: Multiplier applied to the confidence score when soft warnings are present.
#: Reported as a hyper-parameter in the paper.
SOFT_WARNING_PENALTY = 0.8


def field_value(payload: dict, path: str) -> Any:
    """Read a dotted ``path`` out of a model dump, hashably.

    Three shapes appear in the schema:

    * ``"maturity_date"`` - a scalar at the top level.
    * ``"coupon.barrier_pct"`` - a scalar inside a nested object.
    * ``"underlyings.initial_level"`` - one value per element of a list, which is
      returned as a tuple so the whole vector votes as a unit.
    """
    parts = path.split(".")
    current: Any = payload
    for i, part in enumerate(parts):
        if isinstance(current, list):
            rest = ".".join(parts[i:])
            return tuple(field_value(item, rest) for item in current)
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    if isinstance(current, list):
        return tuple(current)
    return current


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, date):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class FieldVote:
    path: str
    value: Any
    agreement: float
    """Share of samples that proposed the winning value, in [0, 1]."""

    n_votes: int
    n_samples: int
    distinct_values: int


@dataclass
class VoteResult:
    votes: dict[str, FieldVote] = field(default_factory=dict)
    n_samples: int = 0

    @property
    def critical_agreements(self) -> dict[str, float]:
        return {
            path: vote.agreement
            for path, vote in self.votes.items()
            if path in CRITICAL_FIELDS
        }

    def confidence(self, *, soft_warnings: int = 0) -> float:
        """Document confidence: the weakest critical field, penalised for warnings."""
        agreements = self.critical_agreements
        if not agreements:
            return 0.0
        score = min(agreements.values())
        if soft_warnings:
            score *= SOFT_WARNING_PENALTY
        return score

    def weakest_field(self) -> tuple[str, float] | None:
        agreements = self.critical_agreements
        if not agreements:
            return None
        path = min(agreements, key=agreements.get)
        return path, agreements[path]


def vote_fields(
    payloads: list[dict], *, paths: tuple[str, ...] = CRITICAL_FIELDS
) -> VoteResult:
    """Majority-vote ``paths`` across ``payloads`` (JSON-mode model dumps).

    Samples that failed to produce any output must be filtered out by the caller;
    a failed call is not a vote for ``None``.
    """
    if not payloads:
        return VoteResult(n_samples=0)

    result = VoteResult(n_samples=len(payloads))
    for path in paths:
        values = [_hashable(field_value(p, path)) for p in payloads]
        counts = Counter(values)
        winner, n_votes = counts.most_common(1)[0]
        result.votes[path] = FieldVote(
            path=path,
            value=winner,
            agreement=n_votes / len(payloads),
            n_votes=n_votes,
            n_samples=len(payloads),
            distinct_values=len(counts),
        )
    return result


def majority_payload(payloads: list[dict]) -> dict | None:
    """The single sample that agrees most often with the field-level majority.

    Field-level voting can produce a combination that no sample proposed, and such
    a Frankenstein parameter set may not satisfy the schema's structural
    invariants.  Choosing the sample closest to the majority keeps the result
    internally consistent while still benefiting from the vote, and is what the
    pipeline prices.
    """
    if not payloads:
        return None
    votes = vote_fields(payloads)
    best, best_score = payloads[0], -1
    for payload in payloads:
        score = sum(
            1
            for path, vote in votes.votes.items()
            if _hashable(field_value(payload, path)) == vote.value
        )
        if score > best_score:
            best, best_score = payload, score
    return best
