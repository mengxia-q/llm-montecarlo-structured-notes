"""Run the four extraction arms (paper: Experimental Setup) on every candidate filing.

    B1    single extraction                        = sample 1 of the k samples
    B2    single extraction + rules                = B1, plus one validator-guided
                                                     retry if B1 has hard errors
    B3    self-consistency vote, no rules          = field-level vote over k samples
    Ours  vote + rules                             = B3, plus one validator-guided
                                                     retry if the vote has hard errors

B1 reuses B3's first sample rather than making an independent call (a paired
design: the arms differ only in what is done with the same draws).  The paper
states this.  Calls run file by file - all samples and retries of one filing
back to back - so that the repeated ~20k-token prefix is served from the
provider's prompt cache.

Every call is appended to ``runs/<run_id>/calls.jsonl`` as soon as it returns,
with the raw response; a re-run skips calls already on disk, so an interrupted
run resumes.  ``arms.jsonl`` holds one record per filing and arm, rebuilt from
the calls each time.

Usage:
    python experiments/07_run_arms.py --run-id stage2 --budget-usd 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import RawNoteTerms, build_prompt, get_provider, to_note_terms
from src.extract.prompt import INSTRUCTIONS, RETRY_PREAMBLE
from src.extract.providers import ExtractionCall
from src.extract.vote import majority_payload, vote_fields
from src.market import initial_level_closes
from src.validate import validate_terms

RESULTS = ROOT / "experiments" / "results"
CLEANED = ROOT / "data" / "cleaned"

#: USD per million tokens, deepseek-flash, off-peak (checked 2026-10-05).
#: Peak hours (UTC Mon-Fri 01-04, 06-10) are billed at twice these rates.
PRICES = {"input_miss": 0.15, "input_hit": 0.003, "output": 0.60}
PEAK_HOURS_UTC = set(range(1, 4)) | set(range(6, 10))


def call_cost(call: dict) -> float:
    hit = call.get("cached_input_tokens") or 0
    miss = max((call.get("input_tokens") or 0) - hit, 0)
    out = (call.get("output_tokens") or 0) + (call.get("thinking_tokens") or 0)
    usd = (miss * PRICES["input_miss"] + hit * PRICES["input_hit"] + out * PRICES["output"]) / 1e6
    when = call.get("called_at")
    if when:
        t = datetime.fromisoformat(when)
        if t.weekday() < 5 and t.hour in PEAK_HOURS_UTC:
            usd *= 2
    return usd


class CallLog:
    """Append-only JSONL of every API call, keyed by (accession, kind)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.calls: dict[tuple[str, str], dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                self.calls[(rec["accession"], rec["kind"])] = rec

    def get(self, accession: str, kind: str) -> dict | None:
        return self.calls.get((accession, kind))

    def add(self, accession: str, kind: str, call: ExtractionCall) -> dict:
        rec = asdict(call)
        rec["raw"] = call.raw.model_dump(mode="json") if call.raw is not None else None
        rec.update(accession=accession, kind=kind)
        rec["cost_usd"] = call_cost(rec)
        with self.lock:
            self.calls[(accession, kind)] = rec
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    @property
    def spent(self) -> float:
        return sum(c.get("cost_usd", 0.0) for c in self.calls.values())


class BudgetExceeded(RuntimeError):
    pass


def evaluate(payload: dict | None) -> dict:
    """Schema conversion + deterministic rules for one candidate parameter set."""
    if payload is None:
        return {"schema_ok": False, "violations": ["no output"], "hard": [], "soft": [], "terms": None}
    terms, violations = to_note_terms(RawNoteTerms.model_validate(payload))
    if terms is None:
        return {
            "schema_ok": False,
            "violations": [str(v) for v in violations],
            "hard": [],
            "soft": [],
            "terms": None,
        }
    try:
        closes = initial_level_closes(terms)
    except Exception:  # noqa: BLE001 - market data is best-effort here
        closes = None
    report = validate_terms(terms, initial_level_closes=closes)
    return {
        "schema_ok": True,
        "violations": [],
        "hard": [str(f) for f in report.hard],
        "soft": [str(f) for f in report.soft],
        "terms": terms.model_dump(mode="json"),
    }


def feedback(check: dict) -> list[str]:
    """Hard findings, including schema-invariant violations, for the retry prompt."""
    return [f"schema: {v}" for v in check["violations"]] + check["hard"]


def needs_retry(check: dict) -> bool:
    return bool(feedback(check)) and check["violations"] != ["no output"]


def run_filing(rec: dict, provider, log: CallLog, args) -> list[dict]:
    acc = rec["accession"]
    text = (CLEANED / f"{acc}_{rec['document']}.txt").read_text()
    prompt = build_prompt(text)

    def call(kind: str, prompt_text: str, seed_index: int) -> dict:
        done = log.get(acc, kind)
        if done is not None and (done["raw"] is not None or done.get("error_kind") == "schema"):
            return done
        if log.spent >= args.budget_usd:
            raise BudgetExceeded(f"budget ${args.budget_usd} reached")
        return log.add(acc, kind, provider.extract(prompt_text, seed_index=seed_index))

    # Sample 0 first so the prefix is cached, then the rest concurrently.
    samples = [call("sample_0", prompt, 0)]
    with ThreadPoolExecutor(max_workers=args.k - 1 or 1) as ex:
        samples += list(ex.map(lambda i: call(f"sample_{i}", prompt, i), range(1, args.k)))

    payloads = [s["raw"] for s in samples if s["raw"] is not None]
    out: list[dict] = []

    def record(arm: str, payload, check, calls, confidence, **extra):
        out.append(
            {
                "accession": acc,
                "arm": arm,
                "payload": payload,
                "schema_ok": check["schema_ok"],
                "violations": check["violations"],
                "hard": check["hard"],
                "soft": check["soft"],
                "confidence": confidence,
                "n_calls": len(calls),
                "cost_usd": sum(c.get("cost_usd", 0.0) for c in calls),
                "latency_s": sum(c.get("latency_s", 0.0) for c in calls),
                **extra,
            }
        )

    # B1 / B2 ---------------------------------------------------------------
    b1 = samples[0]["raw"]
    b1_check = evaluate(b1)
    record("B1", b1, b1_check, samples[:1], None)
    b2, b2_check, b2_calls = b1, b1_check, samples[:1]
    if needs_retry(b1_check):
        retry = call("retry_b2", build_prompt(text, failures=feedback(b1_check)), 100)
        b2_calls = samples[:1] + [retry]
        if retry["raw"] is not None:
            b2, b2_check = retry["raw"], evaluate(retry["raw"])
    record("B2", b2, b2_check, b2_calls, None, retried=len(b2_calls) > 1)

    # B3 / Ours -------------------------------------------------------------
    votes = vote_fields(payloads) if payloads else None
    agreement = {p: v.agreement for p, v in votes.votes.items()} if votes else {}
    b3 = majority_payload(payloads)
    b3_check = evaluate(b3)
    b3_conf = votes.confidence() if votes else 0.0
    record("B3", b3, b3_check, samples, b3_conf, agreement=agreement, n_ok_samples=len(payloads))

    ours, ours_check, ours_calls = b3, b3_check, list(samples)
    if needs_retry(b3_check):
        retry = call("retry_ours", build_prompt(text, failures=feedback(b3_check)), 200)
        ours_calls.append(retry)
        if retry["raw"] is not None:
            ours, ours_check = retry["raw"], evaluate(retry["raw"])
    if not votes or not ours_check["schema_ok"] or ours_check["hard"]:
        ours_conf = 0.0  # still broken after the retry: always manual review
    else:
        ours_conf = votes.confidence(soft_warnings=len(ours_check["soft"]))
    record(
        "Ours", ours, ours_check, ours_calls, ours_conf,
        agreement=agreement, n_ok_samples=len(payloads), retried=len(ours_calls) > len(samples),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidates", default=str(RESULTS / "candidates.jsonl"))
    parser.add_argument("--only", nargs="*", help="restrict to these accessions")
    parser.add_argument("--exclude", default=str(ROOT / "data" / "labels" / "holdout.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--mode", default="strict_tool", choices=["strict_tool", "json_object"])
    parser.add_argument("--thinking", default="on", choices=["on", "off"])
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--file-workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = RESULTS / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = [json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l]
    if args.only:
        candidates = [c for c in candidates if c["accession"] in set(args.only)]
    exclude_path = Path(args.exclude)
    if exclude_path.exists():
        held = set(json.loads(exclude_path.read_text()).get("accessions", []))
        candidates = [c for c in candidates if c["accession"] not in held]

    provider = get_provider(
        "deepseek",
        model=args.model,
        temperature=args.temperature,
        mode=args.mode,
        thinking=args.thinking == "on",
        reasoning_effort=args.reasoning_effort,
    )
    probe = provider.request_kwargs("", args.temperature)
    config = {
        "model": args.model,
        "mode": args.mode,
        "endpoint": provider.endpoint,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "k": args.k,
        "b1_is_sample_0": True,
        "prompt_sha256": hashlib.sha256((INSTRUCTIONS + RETRY_PREAMBLE).encode()).hexdigest(),
        "request_template_sha256": hashlib.sha256(
            json.dumps(probe, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "n_filings": len(candidates),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ("model", "mode", "thinking", "temperature", "k", "prompt_sha256"):
            if old.get(key) != config[key]:
                print(f"config mismatch on {key}: {old.get(key)} != {config[key]}", file=sys.stderr)
                return 2
        # A resumed run may cover more filings than the invocation that created it.
        if config["n_filings"] > old.get("n_filings", 0):
            old["n_filings"] = config["n_filings"]
            config_path.write_text(json.dumps(old, indent=2))
    else:
        config_path.write_text(json.dumps(config, indent=2))

    log = CallLog(run_dir / "calls.jsonl")
    print(f"{len(candidates)} filings, {len(log.calls)} calls on disk, spent ${log.spent:.4f}")

    arms: list[dict] = []
    failed: list[str] = []

    def one(rec):
        try:
            res = run_filing(rec, provider, log, args)
            conf = next(r["confidence"] for r in res if r["arm"] == "Ours")
            print(f"  {rec['accession']} {rec.get('issuer_stratum', '')[:24]:24s} "
                  f"ours conf {conf:.2f}  spent ${log.spent:.3f}", flush=True)
            return res
        except BudgetExceeded as exc:
            print(f"  {rec['accession']}: {exc}", flush=True)
            failed.append(rec["accession"])
            return []

    with ThreadPoolExecutor(max_workers=args.file_workers) as ex:
        for res in ex.map(one, candidates):
            arms.extend(res)

    order = {c["accession"]: i for i, c in enumerate(candidates)}
    arms.sort(key=lambda r: (order[r["accession"]], ["B1", "B2", "B3", "Ours"].index(r["arm"])))
    (run_dir / "arms.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in arms))
    print(f"wrote {len(arms)} arm records; total spent ${log.spent:.4f}; "
          f"{len(failed)} filings stopped by budget")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
