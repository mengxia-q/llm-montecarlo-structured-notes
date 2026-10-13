# llm-montecarlo-structured-notes

Code, data and model outputs for *Combining LLM-Based Term Extraction and Monte
Carlo Valuation for Structured Notes: A Pricing-Aware Evaluation*.

An LLM only translates an SEC 424B2 pricing supplement into a pricing schema; a
tested Monte Carlo template computes the value. Extraction quality is measured
in dollars: every extraction and its reference label are priced with common
random numbers, so the price deviation isolates extraction error.

```
424B2 HTML ─► clean ─► LLM extraction (k = 5 samples) ─► field-level vote
                                                              │
                          Monte Carlo valuation ◄── business rules (+ one retry)
                                    │
                 Δ = V(extracted) − V(reference), per $1,000 face
```

## Setup

```bash
python3.11 -m venv ~/.venvs/snllm
~/.venvs/snllm/bin/pip install -r requirements.txt
export DEEPSEEK_API_KEY=...        # only needed to re-run the extraction
export OPENAI_API_KEY=... OPENAI_BASE_URL=...   # Qwen3.8-Flash, OpenAI-compatible endpoint
```

**Run every script with this environment's Python from the repository root.**
A Python without `yfinance` cannot load market data that is missing from the
cache, and the evaluation then records those filings as unpriceable and
overwrites the results.

API keys are read only from environment variables (`DEEPSEEK_API_KEY`;
`OPENAI_API_KEY` and `OPENAI_BASE_URL` for Qwen; `GEMINI_API_KEY` for the
pilot); no key is stored in the repository.

```bash
~/.venvs/snllm/bin/python -m pytest tests/ -q   # 69 tests: payoff rules, engine, rules, schema, cleaner
```

## Reproducing the paper

The filings, market data and model outputs are all in the repository, so every
table, figure and number can be rebuilt offline, without calling the API.

```bash
PY=~/.venvs/snllm/bin/python

# Evaluation, dataset and cost statistics, and paper/numbers.tex
$PY experiments/14_make_gold.py --refresh

# Per-field sensitivity and seed stability (figure 3); not part of --refresh,
# so re-run it after any change under src/pricing/
$PY experiments/10_sensitivity.py --run-id stage2_nothink

# Monte Carlo convergence (the "simulation noise" paragraph)
$PY experiments/08_convergence.py

# Figures 2-3 (the script also draws a coverage figure no longer in the paper)
$PY experiments/12_figures.py

# Revision: robustness to the market assumptions, per-sample calibration
$PY experiments/19_model_sensitivity.py --run-ids stage2 stage2_nothink
$PY experiments/21_calibration.py --run-ids stage2 stage2_rep1 stage2_rep2 stage2_rep3 \
    stage2_nothink stage2_nothink_rep1 stage2_nothink_rep2 stage2_nothink_rep3 \
    qwen_think qwen_nothink stress_think stress_nothink
for r in experiments/results/runs/*; do $PY experiments/09_evaluate.py --run-id ${r##*/}; done
$PY experiments/13_paper_numbers.py
```

`paper/numbers.tex` holds every number the paper quotes, as a LaTeX macro, and
`paper/figures/` the figures. Both are generated; compare them with the paper
rather than editing them. `numbers.tex` defines more macros than the paper
uses (the full arm × mode × subset grid).

To re-run the extraction itself (the paper's runs cost $0.91 with reasoning and
$0.31 without, at off-peak rates):

```bash
$PY experiments/07_run_arms.py --run-id <new-id> --thinking on
$PY experiments/07_run_arms.py --run-id <new-id>_nothink --thinking off
```

`07_run_arms.py` appends each call to `runs/<run_id>/calls.jsonl` as soon as it
returns and skips calls already on disk, so an interrupted run resumes; the
committed runs `stage2` (reasoning on) and `stage2_nothink` (reasoning off) are
complete, so use a new run id. The paper's runs were made on 2026-10-06 against
the API alias `deepseek-flash` (DeepSeek-V4.1-Flash), at temperature 0.7.

Runs added in the revision (2026-10-13):

| Run ids | What |
|---|---|
| `stage2_rep1..3`, `stage2_nothink_rep1..3` | Three further independent sets of k = 5 DeepSeek samples per filing |
| `stage2_b0`, `stage2_nothink_b0`, `qwen_*_b0` | B0: one extraction without constrained decoding (`--mode json_object --k 1`) |
| `qwen_think`, `qwen_nothink` | Qwen3.8-Flash (`--provider qwen`), constrained by a strict JSON-schema response format |
| `stress_think`, `stress_nothink` | DeepSeek on tables rendered without rowspan expansion (`--cleaned-dir data/cleaned_norowspan`, built by `20_clean_norowspan.py`) |

`01_build_candidate_pool.py` rebuilds the candidate pool from EDGAR; see its
docstring for the passes used. It needs a contact e-mail for the SEC User-Agent.

## Repository layout

| Path | Contents |
|---|---|
| `src/schema.py` | `NoteTerms`: the pricing schema, its invariants and the nine `CRITICAL_FIELDS` |
| `src/crawl/` | EDGAR full-text search per issuer, rate-limited download, scope screening |
| `src/clean/` | HTML to text, with tables rendered as Markdown and running headers removed |
| `src/extract/` | Prompt, relaxed model-output schema, providers (DeepSeek, Qwen; Gemini for the pilot), field-level voting and confidence |
| `src/validate/` | Hard and soft business rules |
| `src/market/` | Ticker registry, NYSE calendar, FRED zero curve, volatility, correlation and dividend estimation |
| `src/pricing/` | Common random numbers (scrambled Sobol'), GBM paths, payoff rules, valuation engine |
| `src/eval/` | Paired valuation and the price deviation Δ |
| `experiments/` | Numbered scripts, below; outputs in `experiments/results/` |
| `data/` | Filings, market data and labels, below |
| `paper/` | Generated `numbers.tex` and figures |
| `tests/` | Unit tests |

### Experiment scripts

| Script | Role |
|---|---|
| `01_build_candidate_pool.py` | Stratified EDGAR search, screening, cleaning → `results/candidates.jsonl`, `results/pool_exclusions.json` |
| `02_vertical_slice.py` | Pilot run before any labels existed; defines the perturbations reused by `10` |
| `06_check_prelabels.py` | Checks every prelabel: verbatim evidence quotes, schema, business rules, initial level against the market close |
| `07_run_arms.py` | Runs the four arms (B1, B2, B3, Ours) → `results/runs/<run_id>/` |
| `08_convergence.py` | Monte Carlo convergence and variance reduction → `results/convergence.json` |
| `09_evaluate.py` | Scores every arm against the labels → `eval_rows.jsonl`, `summary.json` |
| `10_sensitivity.py` | Per-field perturbation sensitivity and seed stability → `sensitivity.json` |
| `11_stats.py` | Dataset statistics (Table 1), cost and latency (Table 3) → `results/stats.json` |
| `12_figures.py` | Figures 2–4 → `paper/figures/` |
| `13_paper_numbers.py` | Every number in the paper → `paper/numbers.tex` |
| `14_make_gold.py` | Promotes a verified prelabel to gold; `--refresh` reruns 09, 11 and 13 |
| `19_model_sensitivity.py` | Ranking and Δ under 10 alternative market assumptions → `results/model_sensitivity.json` |
| `20_clean_norowspan.py` | Stress-test input: the filings cleaned without rowspan expansion → `data/cleaned_norowspan/` |
| `21_calibration.py` | Every individual sample scored as B1, with pre-vote agreement → `results/calibration.json` |
| `22_ingest_rounds34.py` | Third (independent annotation) and fourth (verification) labelling rounds → gold labels, `data/labels/round34_report.json` |

Each run directory holds `calls.jsonl` (every API call with its raw response,
token usage and latency), `arms.jsonl` (the four arms' outputs per filing),
`config.json`, `eval_rows.jsonl` and `summary.json`.

## Data

| Path | Contents |
|---|---|
| `data/raw/` | The original 424B2 HTML of the 53 candidate filings (`experiments/results/candidates.jsonl` gives each one's EDGAR URL) |
| `data/cleaned/` | Their cleaned text. Evidence line numbers in the labels refer to these files |
| `data/market/` | Daily closes and ETF distributions, and FRED Treasury yields, as used in the paper |
| `data/labels/prelabel/` | One label per filing by Claude Opus 5.5 (one isolated agent per filing), with a verbatim quote and line span for every field; prompt in `experiments/prelabel_prompt.md` |
| `data/labels/gold/` | All 50 labels, checked by humans against the filing: 45 verified field by field from the prelabel, 5 annotated from scratch; `provenance` records the round, role, minutes and changed fields |
| `data/labels/exclusions.json` | Three filings dropped after labelling (buffered downside) |
| `data/labels/round2/round2_report.json` | Second verification round: minutes per file, seeded-error detection, agreement on the files both verifiers checked |
| `data/labels/round34_report.json` | Third round (5 independent annotations, compared field by field with the prelabel) and fourth round (22 verifications) |
| `data/labels/LABELLING_GUIDELINE.md` | Labelling rules, scope and verification protocol |

The evaluation uses the 50 filings that are not excluded; every one has a
human-checked gold label.

**Market data notice.** Index and ETF closes and ETF distributions were
downloaded from Yahoo Finance through `yfinance` and are included only so that
the paper's prices can be reproduced exactly; Yahoo revises history, and a
fresh download can shift the estimated volatilities and dividend yields. These
series remain subject to Yahoo's terms of use and are not covered by this
repository's licences. `treasury_cmt.csv` is from FRED (Federal Reserve Bank of
St. Louis). If a file is missing, `src/market/data.py` downloads it again.

## Model and pricing assumptions

* Multi-asset GBM under the risk-neutral measure. Volatility and correlation
  from the 252 trading days ending on the pricing date; constant dividend
  yield per underlying from trailing distributions; forward rates from the
  FRED Treasury zero curve. No issuer credit spread: it would lower every
  price, but it is identical across a paired valuation and cancels in Δ.
* Paths start at the pricing-date close and performance is divided by the
  *extracted* initial level, so a wrong initial level has a price.
* Draws are made once at a fixed `(n_paths, max_steps, max_assets)` shape and
  sliced, and the Sobol' dimension is fixed per filing, so that a wrong
  observation-date count does not shift the random sequence.
* 2^17 scrambled Sobol' paths (Sobol' point sets are balanced only at powers of two).
* In scope: contingent-coupon autocallables on equity indices and ETFs, single
  or worst-of. Single stocks (discrete dividends), buffered downside and daily
  knock-in monitoring are out of scope.

## Sources

* SEC EDGAR full-text search and archives.
* FRED constant-maturity Treasury yields.
* Yahoo Finance via `yfinance`, for index and ETF closes and ETF distributions.

## Licence

* **Code** (`src/`, `experiments/`, `tests/`): MIT, see `LICENSE`.
* **Labels, model outputs and derived results** (`data/labels/`,
  `experiments/results/`, `paper/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Please cite the paper when you use them.
* **Third-party data**, which keeps its own terms: the SEC filings in
  `data/raw/` and `data/cleaned/` are public records from EDGAR; the market
  data in `data/market/` is covered by the notice above.
