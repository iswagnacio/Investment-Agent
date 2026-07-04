# Investment — AI Stock Evaluation Pipeline

A multi-agent pipeline that evaluates stocks with LLMs and records structured,
diffable verdicts over time. It produces **qualitative judgment** — signal,
conviction, thesis, risks, and "what would change my mind" — not trade
construction: no price targets, options strategies, or P&L.

See `ROADMAP.md` for the full architecture assessment and the four completed
build phases.

## How it works

```
                        ┌─ calculators/ ── deterministic data (yfinance, news APIs)
gather ─ score ─ synthesize ─ record ─ render
   │        │         │          │        └─ human report (a view of the record)
   │        │         │          └─ data/runs/<date>/<ticker>.json  (the product)
   │        │         └─ lead agent: two-stage LLM synthesis
   │        └─ scoring.py: deterministic 0-100 blend, registry weights
   └─ agents/: technical, fundamental, sentiment specialists (structured JSON)
      + optional investor perspective and registry-declared custom assessments
```

Design principles (enforced throughout):

- **Calculator/agent pairing** — each domain has a data-only calculator
  (`investment/calculators/`) and an LLM agent (`investment/agents/`) with a
  uniform `analyze(ticker, analysis_type)` surface.
- **Structure first, prose second** — every evaluation builds a persisted
  `AnalysisRecord` (`investment/contracts.py`) *before* any report is rendered
  (`investment/reporting.py`), the same way the distiller writes
  `framework.json` before `SKILL.md`.
- **Offline/runtime split** — expensive authoring (perspective distillation,
  X-post fetching) happens in `investment/offline/`; the runtime pipeline only
  reads the caches it produces.
- **Trust is earned** — perspectives and custom assessments influence the blend
  only through quality gates, confidence shrinkage, and a maturity ramp; the
  three core domains always keep the majority.

## Setup

```bash
conda activate invest              # Python 3.11 env with pinned deps
pip install -e ".[dev]"            # optional: install as a package + pytest
```

Required in `.env` (repo root):

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | all LLM analysis |
| `TAVILY_API_KEY` | perspective distillation (web mode) |
| `FINNHUB_API_KEY`, `NEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY` | extra news sources (optional; Yahoo works without) |
| `X_BEARER_TOKEN` | fetching X posts (`X_fetch.py`) |

## Usage

**Analyze a stock** (full report + persisted record):

```bash
python lead_agent.py MU
python lead_agent.py MU --perspective "charlie-munger" --perspective-weight 0.15
python lead_agent.py MU AVGO --quick        # scores/signal only, no synthesis
```

**Scheduled evaluation** (headless; this is what cron/launchd should invoke):

```bash
python runner.py --dry-run                  # preview universe + config, free
python runner.py                            # evaluate universe from holdings.csv / runner.toml
python runner.py --tickers MU AVGO          # explicit universe
```

Each run writes `data/runs/<YYYY-MM-DD>/`:
- `<TICKER>.json` — one `AnalysisRecord` per ticker (versioned in git)
- `digest.md` — **only what changed** vs. each ticker's previous record:
  signal flips, ±10-point score moves, new tickers, failures
- `run.log` — full log (gitignored)

Exit codes: `0` all evaluated, `1` some failed, `2` nothing ran. Configure the
universe/perspective/parallelism in `runner.toml`; the schedule itself lives
outside the repo (cron, launchd, or a Claude Code routine).

**Distill an investor perspective** (offline, once per investor):

```bash
python distill_investor.py "Charlie Munger" --review
python distill_investor.py --posts serenity_posts.json       # from fetched X posts
python distill_investor.py "Charlie Munger" --from-brief     # re-extract after prompt changes, no re-fetch
```

Writes `data/perspectives_cache/<key>/` (`framework.json` + `SKILL.md` +
`metadata.json` + cached `source_brief.txt`). Vet the SKILL.md before relying
on it — the quality gate shrinks a thin perspective's weight automatically.

**Manage the universe** — `holdings.csv` tracks *which* tickers are held or
watched (status, date, notes). It is not a P&L tracker.

## Custom assessments (agent autonomy)

Adding an evaluation lens is a config entry + an artifact — no code changes:

1. Author a methodology artifact into a workspace:
   `python distill_investor.py "Some Source" --workspace`, or write the
   `framework.json` shape directly into `registry.ensure_workspace("<key>")`
   (`data/workspaces/<key>/`).
2. Register it in `registry.toml` with a weight cap and a permissions manifest
   (which data sources it may see, its per-run LLM budget).

Its effective weight = `min(weight, max_weight, 0.25)` × artifact confidence
(High 1.0 / Medium 0.7 / Low 0.4) × maturity (25% on first run → 100% after 20
recorded runs). All post-blend assessments together are capped at 40%.
`registry.toml` has a commented example.

## Layout

```
investment/            the package
  config.py            paths (repo-root anchored), models, weights, helpers
  contracts.py         AnalysisRecord — the persisted result schema
  scoring.py           Signal + deterministic ScoreCalculator
  registry.py/.toml    core weights + custom assessments + permissions
  reporting.py         renders reports from records (never computes)
  runner.py/.toml      headless scheduled runs + change digest
  universe.py          holdings.csv universe manager
  agents/              lead orchestrator + specialist LLM agents
  calculators/         data-only fetchers/calculators (no LLM)
  offline/             distiller + X post fetcher (authoring tools)
data/                  caches (gitignored) + runs/ (versioned record history)
tests/                 pytest suite — pure functions, no network (89 tests)
scripts/               live-API smoke scripts (spend tokens; run deliberately)
lead_agent.py, runner.py, distill_investor.py, X_fetch.py
                       root entrypoint shims for the commands above
```

## Testing

```bash
python -m pytest       # offline unit suite; no API keys, no network
```
