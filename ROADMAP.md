# Code Assessment & Forward Roadmap

*Assessed 2026-07-03. Covers the state of the repo as of commit `6ffcfa3` plus the
uncommitted `distill_investor.py` prompt rework.*

The direction: **AI evaluates the stock.** Evaluations run on a user-given schedule;
agents get permission and environments to create their own assessments as needed;
profit calculations (price targets, options strategies, P&L) are retired from the
core signal. This document records what the code does well, what blocks that
direction, and a phased plan that preserves the existing design.

---

## Part 1 — Assessment

### What the design gets right (preserve these)

1. **The calculator/agent pairing.** Each domain is split into a data-only
   calculator (`technical.py`, `fundamental.py`, `sentiment.py` — no LLM, cacheable,
   deterministic) and an LLM agent (`*_agent.py`) with a uniform surface:
   `__init__(model)` + `analyze(ticker, analysis_type)`. This is a real contract even
   though it is only enforced by convention, and it is why adding the fourth agent
   (perspective) was cheap.

2. **The offline/runtime split with single-writer caches.**
   `distill_investor.py` is the only writer of `data/perspectives_cache/`;
   `perspective_agent.py` only reads and raises `PerspectiveNotCachedError` on a miss.
   Expensive LLM work happens offline; the runtime pipeline stays fast and cheap.
   The `framework.json` (machine) + `SKILL.md` (human render) pattern is the right
   shape for every future agent-authored artifact.

3. **Structure first, prose second.** Specialists return structured JSON
   (`for_synthesis`), the lead agent scores deterministically
   (`ScoreCalculator`), and only then does LLM synthesis run. Prompts are the
   variable part; the scoring math is auditable.

4. **Defensive orchestration.** Graceful degradation when a specialist fails to
   import/init, ticker-mismatch validation with one retry, JSON-extraction
   fallbacks, neutral 50-scores plus confidence penalties for missing data, and
   distillation-quality gates that *shrink the perspective weight* rather than
   trusting it blindly (`_effective_perspective_weight`). This "cap the influence of
   low-trust inputs" idea generalizes directly to future self-created agents.

### What blocks the future direction (fix these)

| # | Problem | Where | Why it blocks the plan |
|---|---------|-------|------------------------|
| 1 | Output is a formatted ASCII report string; nothing machine-readable is persisted per run | `lead_agent.analyze()` returns a `str`; `AnalysisResult` dataclass exists but is never actually the interchange format | A scheduled run has nothing durable to write, diff against yesterday, or alert on. **This is the single biggest blocker to automation.** |
| 2 | Profit-calculation code is load-bearing | `OptionsStrategy` enum, `determine_options_strategy`, `calculate_price_targets`, report header %, `get_quick_recommendation`, `compare_stocks`, `portfolio.py` cost-basis math | Retiring it touches the core dataclass, the report template, and every CLI path — it must be an explicit phase, not a drive-by edit |
| 3 | No packaging or dependency manifest | no `pyproject.toml` / `requirements.txt`; deps (langchain-anthropic, yfinance, pandas, tavily, dotenv, …) are implicit | A cron/launchd job or cloud routine can't reliably reconstruct the environment |
| 4 | Hardcoded **relative** paths | `data/perspectives_cache` (2 files), `holdings.csv`, `data/*_cache` | Break the moment the entrypoint runs from a different cwd — which is exactly what schedulers do |
| 5 | `print()`-based, emoji-decorated logging | throughout `lead_agent.py`, `distill_investor.py` | Headless runs need log levels, run IDs, and files; today's output is only usable interactively |
| 6 | Config scattered and duplicated | weights 0.35/0.45/0.20 live in `ScoreCalculator.calculate_overall_score` **and** are re-stated in `calculate_scores`; model names hardcoded in 8 places; `MAX_PW=0.25` inline | Dynamic per-agent weighting (Phase 3) is impossible while weights are literals in two places |
| 7 | Convention-coupled duplicates | `normalize_key` in `distill_investor.py` + `perspective_agent.py` ("must match" comment); `_strip_fences` ×2; two unrelated `YahooFinanceFetcher` classes (fundamental.py, sentiment.py) | Silent divergence risk; the comment is the only enforcement |
| 8 | Monolith files | `lead_agent.py` 2,061 lines (models + scoring + prompts + orchestration + report + CLI); `fundamental.py` 1,572 lines (fetchers + cache + ratios) | Every future feature lands in the same two files; merge/review cost compounds |
| 9 | Tests are live-API demo scripts | `test_lead.py`, `test_funda.py`, `test_senti.py` hit real endpoints and spend tokens; not pytest | No CI, no regression safety for the refactors below |
| 10 | Fully sequential pipeline | `gather_data` runs ~4–6 LLM calls + network fetches serially per ticker | Screening a universe on a schedule multiplies this; needs at least per-ticker parallelism eventually |
| 11 | Dead code | `perspective_agent.py:346-349` — vestigial `if False else` in the CLI block | Trivial, but it's in the file other agents are told to mirror |
| 12 | `data/` entirely gitignored | `.gitignore` | Fine for caches; wrong for future run history and agent workspaces, which are the system's memory |

**Verdict:** the *design* is sound and worth keeping — the pipeline stages, the
agent contract, and the offline-distill/runtime-apply split are exactly the bones a
scheduled multi-agent system needs. What's missing is the *operational* layer:
structured persisted results, real config, packaging, and headless execution. None
of that requires abandoning the current structure; it requires formalizing what is
already convention.

---

## Part 2 — Roadmap

Phases are ordered by dependency: each one unblocks the next. Within a phase, work
follows the existing patterns (calculator/agent pairing, single-writer caches,
structure-then-render).

### Phase 0 — Hygiene (small, do first) — ✅ done 2026-07-03

*Delivered: `pyproject.toml` (pinned to the `invest` conda env), `config.py`
(root-anchored paths, models, weights, shared `normalize_key`/`strip_fences`,
run-ID logging helpers), all modules wired to it, dead CLI branch removed,
43 pytest unit tests in `tests/`, old live-API scripts moved to `scripts/`.
Print→logging migration is deliberately incremental (as files get touched).*

- **`pyproject.toml`** with pinned deps; the repo becomes installable
  (`pip install -e .`). Keep the flat module names importable during migration.
- **`config.py`** — single source of truth:
  - all paths anchored to the repo root via `Path(__file__).parent`
    (fixes blocker #4);
  - model names (`STRONG_MODEL`, `CHEAP_MODEL` — the distiller already names them,
    promote that convention repo-wide);
  - blend weights and `MAX_PERSPECTIVE_WEIGHT` (fixes #6);
  - one `normalize_key` / `strip_fences` home (fixes #7).
- **`logging`** module setup with a per-run ID; `print` calls migrate as files are
  touched, not in one sweep.
- **pytest** skeleton: unit tests for `ScoreCalculator`, `extract_json_from_text`,
  `normalize_key`, quality gates — all pure functions, no network. Existing
  `test_*.py` scripts move to `scripts/` as smoke tests.
- Delete the dead CLI branch in `perspective_agent.py` (#11).

### Phase 1 — Structured results become the contract — ✅ done 2026-07-03

*Delivered: `contracts.py` (`AnalysisRecord` with save/load/`latest(before_day=…)`
for Phase 2 diffing), `reporting.py` (report rendered from the record only),
`lead_agent.analyze()` now builds → persists → renders; `OptionsStrategy`,
`determine_options_strategy`, `calculate_price_targets`, and the unused
`AnalysisResult` removed; synthesis prompt asks for verdict/conviction/
"what would change my mind" instead of entry/stop/options; `portfolio.py` is a
universe manager (`get_universe()` added, cost/P&L reporting removed, CSV shape
unchanged); `.gitignore` now versions `data/runs/`. Bonus distiller work:
source briefs are cached and `--from-brief` re-extracts with an improved prompt
without re-fetching sources; metadata carries `schema_version`.*

*The prerequisite for scheduling. Mirrors the distiller's structure-then-render pattern.*

- **`AnalysisRecord`** (in a new `contracts.py`): the persisted, versioned JSON
  result of one evaluation — ticker, timestamp, per-agent signals
  (score/verdict/narrative/metadata), blend weights actually used, overall
  signal + confidence, data-freshness notes, model IDs. `lead_agent.analyze()`
  builds the record **first**, writes it to `data/runs/<YYYY-MM-DD>/<ticker>.json`,
  and only then renders the human report from it (exactly as `framework.json` →
  `SKILL.md`).
- **Retire profit calculations** (the explicit break with the old scope):
  - drop `OptionsStrategy`, `determine_options_strategy`, `calculate_price_targets`
    and the price-target block of the report;
  - `AnalysisRecord` carries *thesis, signal, confidence, key risks, catalysts* —
    qualitative judgment, not trade construction;
  - `portfolio.py` / `portfolio_agent.py` shrink to a **universe manager**:
    `holdings.csv` remains the list of tickers to evaluate (held vs. watch), but
    cost-basis/P&L summary goes away.
- `data/runs/` gets an exemption from `.gitignore` (or a retention policy) — run
  history is the system's memory and the input to Phase 3's track-record weighting.

### Phase 2 — Scheduled evaluation — ✅ done 2026-07-03

*Delivered: `runner.py` (headless, cwd-independent, `--config/--tickers/--dry-run`,
exit codes 0/1/2, logs to `data/runs/<date>/run.log`), `runner.toml` sample config
(universe / perspective / parallelism; schedule itself stays in cron/launchd or a
Claude routine), digest stage writing `digest.md` per run — signal changes,
±10-point moves, new tickers, failures; unchanged tickers get one line.
`lead_agent.analyze_to_record()` added as the headless API; also fixed
import-time binding of `config.RUNS_DIR` defaults in contracts.py.*

- **`runner.py`** — the headless entrypoint:
  `python runner.py --config schedule.yaml`. Reads a small config: the universe
  (from `holdings.csv` or explicit tickers), which specialists and perspectives to
  apply, weights, output dir. No interactive prints; logs to
  `data/runs/<date>/run.log`; nonzero exit on failure so any scheduler can alert.
- **Diff/digest stage**: after a run, compare each record to the previous one for
  that ticker and emit `digest.md` listing **signal changes only** (Buy→Hold, new
  red flags, confidence collapses). A schedule that re-prints the same report daily
  is noise; the diff is the product.
- The schedule itself stays **outside** the codebase: cron/launchd or a Claude Code
  scheduled routine invokes `runner.py`. The repo's job is to be safely invokable,
  not to daemonize.
- Cheap parallelism where safe: per-ticker `ThreadPoolExecutor` in the runner
  (the per-ticker pipeline stays sequential and debuggable).

### Phase 3 — Agent autonomy: self-created assessments — ✅ done 2026-07-03

*Delivered: `registry.py` + `registry.toml` (core weights moved to the registry;
`[agents.<key>]` entries with permission manifests — data_sources, max_llm_calls,
workspace_write), per-agent workspaces under `data/workspaces/<key>/` holding
framework.json + SKILL.md + metadata.json artifacts (authored via
`distill_investor.py --workspace` or written directly by an agent), applied at
runtime by the existing perspective machinery pointed at the workspaces dir.
Trust-gated weighting: min(weight, max_weight, 0.25) × artifact confidence
(1.0/0.7/0.4) × maturity ramp (25% → 100% over 20 recorded runs), with all
post-blend assessments jointly capped at 40% so the core keeps the majority.
`registry.appearances()` counts run-history presence — the hook where
correctness-based track-record scoring lands next.*

*Generalizes what the perspective system already proves: an agent can author its own
methodology offline, and the lead agent applies it at runtime with trust-gated weight.*

- **Formalize the agent contract** (`contracts.py`): a small protocol —
  `name`, `analyze(ticker, context) -> AssessmentSignal`
  (score 0–100, verdict, narrative, metadata incl. self-reported confidence).
  All four existing agents already satisfy it modulo naming; adapt, don't rewrite.
- **Registry + discovery**: the lead agent discovers assessments the way
  `list_available_perspectives()` already scans the perspective cache — a
  `registry.yaml` (or directory scan) listing enabled agents, their weights, and
  their trust status. Adding an assessment = adding an entry, not editing
  `lead_agent.py`.
- **Per-agent workspaces**: `data/workspaces/<agent_key>/` — each agent's own
  environment to persist notes, intermediate caches, and (crucially)
  **self-authored methodology artifacts** in the `framework.json` + `SKILL.md` +
  `metadata.json` shape. The distiller pattern becomes the *general* mechanism by
  which any agent creates a new assessment for itself.
- **Permissions manifest** per agent (in the registry entry): which data sources it
  may call, token/call budget per run, and whether it may write outside its
  workspace. The lead agent enforces this the same way it already enforces
  perspective quality: violations or missing gates → weight shrunk or agent skipped.
- **Dynamic, trust-gated weighting**: replace the hardcoded 0.35/0.45/0.20 with
  registry weights. New self-created assessments enter at a low capped weight
  (like `MAX_PW=0.25` today) and can earn weight only from **track record** —
  which Phase 1's run history makes measurable. Signal correctness over time, not
  profit, is the yardstick.

### Phase 4 — Structural migration (mechanical, last)

Target layout — same design, real packages:

```
investment/
  config.py               # Phase 0
  contracts.py            # AssessmentSignal, AnalysisRecord, agent protocol
  registry.yaml           # Phase 3: enabled agents, weights, permissions
  runner.py               # Phase 2 headless entrypoint
  agents/
    lead.py               # orchestration + synthesis prompts only
    technical.py          # from technical_agent.py
    fundamental.py        # from fundamental_agent.py
    sentiment.py          # from sentiment_agent.py
    perspective.py        # from perspective_agent.py
  calculators/
    technical.py          # data-only, unchanged pattern
    fundamental.py        #   (split fetchers / cache / ratios internally)
    sentiment.py
  scoring.py              # ScoreCalculator out of lead_agent.py
  reporting.py            # report rendering out of lead_agent.py
  offline/
    distill_investor.py
    x_fetch.py
data/
  runs/                   # Phase 1: persisted AnalysisRecords + digests
  workspaces/             # Phase 3: per-agent environments
  perspectives_cache/     # unchanged
  fundamental_cache/  news_cache/  x_cache/
tests/
scripts/                  # the old live-API smoke scripts
```

The migration is renames plus import updates — the calculator/agent pairing, the
single-writer cache discipline, and the structure-then-render flow all survive
unchanged. Do it last so it's mechanical over already-tested code, not a rewrite.

### Sequencing summary

| Phase | Unlocks | Risk |
|-------|---------|------|
| 0. Hygiene | reproducible env, testability, one config | trivial |
| 1. AnalysisRecord + retire profit calc | anything automated; the new qualitative scope | medium — touches report & CLI paths |
| 2. runner.py + diff digest | user-given schedules | low, additive |
| 3. Contract + registry + workspaces + trust weights | agents creating their own assessments | medium — new surface, but modeled on the proven perspective system |
| 4. Package layout | long-term maintainability | mechanical |
