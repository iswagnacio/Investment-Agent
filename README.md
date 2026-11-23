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

Use **Python 3.11** for the verified setup. Run the following from the repository
root in a new virtual environment; an existing conda environment is not required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
cp -n .env.example .env
```

If you already have a `.env`, keep it and add any missing variables instead of
overwriting it. Fill in `ANTHROPIC_API_KEY` locally. Exported environment variables
also work and take precedence over `.env`. Never commit real credentials.

`requirements.txt` installs the local package and Streamlit web extra using the
compatible dependency versions in `pyproject.toml`; keep that manifest as the source of truth. To install
the optional test dependencies, run `python -m pip install -e ".[dev]"`.

Conda alternative: `conda create -n invest-500d python=3.11`, then
`conda activate invest-500d` and follow the pip and `.env` steps above.

The credential template lists:

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | Required for research, portfolio queries, and specialist agents |
| `FINNHUB_API_KEY`, `NEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY` | Optional extra news sources; Yahoo requires no API key, but still needs network access |
| `TAVILY_API_KEY` | Optional offline perspective distillation in web mode |
| `X_BEARER_TOKEN` | Optional offline X-post fetching (`X_fetch.py`) |

Research requires access to Yahoo Finance and Anthropic. The configured models are
`claude-sonnet-4-5` (research) and `claude-haiku-4-5` (portfolio); your Anthropic
account must have access and available credit. Optional news services have their
own quotas. A successful install or offline test does not verify live API access.

The local web interface includes short-term portfolio conversation memory.
For a teammate running the presentation demo locally, see
[DEMO_HANDOFF.md](DEMO_HANDOFF.md). For Community Cloud source packaging,
required secrets, and manual publication steps, see
[DEPLOYMENT.md](DEPLOYMENT.md). The hosting account setup and public URL check
are still pending.

## Local web interface

After setup, start the app from the repository root:

```bash
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open the Local URL printed by Streamlit (normally `http://127.0.0.1:8501`). The
initial screen needs no credentials and makes no model calls. Set
`ANTHROPIC_API_KEY` in `.env` or the environment before submitting requests.
Restart the Streamlit server after changing credentials; existing agent modules
load `.env` when they are first imported.
If you installed only the base package, add the UI with
`python -m pip install -e ".[web]"`.

- **Stock Research:** enter `AAPL` and click **Run research**. This calls the
  existing `GeneralAnalystAgent.analyze()` and displays its scores, confidence,
  signal, generation time, final analysis, and full consolidated report where
  available. Pipeline notes remain visible when sources are missing. This uses
  paid model calls. The web demo validates source availability and dates before
  scoring/final synthesis, then displays the report without saving a shared run
  record. CLI research continues to save dated records by default.
- **Portfolio Assistant:** ask “What's in my portfolio?” or “What's on my
  watchlist?” and click **Ask assistant**. The existing ReAct agent receives a
  temporary synthetic CSV with AAPL/MSFT holdings and an NVDA watchlist item.
  The agent has four read tools; additions and edits are unavailable, and its
  underlying portfolio manager also rejects writes. Its answer and actual tool
  observations are displayed without internal reasoning traces. The temporary
  CSV is removed after success or failure.

The workflows are independent. Stock research remains a single-request workflow.
The portfolio assistant retains up to **six complete user/assistant exchanges per
Streamlit session** and passes those messages into the existing ReAct prompt for
follow-ups. Each retained message is capped at 2,000 characters; the complete
serialized history is capped at 12,000 characters. Oldest complete exchanges are
dropped first, and shortened messages are marked `[truncated]`. Failed requests
and incomplete agent runs do not enter the retained conversation. The latest
answer remains visible in full; the shorter version is used for retained context.

Click **Reset conversation** to remove portfolio context, visible answers, tool
results, errors, and the question field. The next question starts without prior
messages. Separate sessions have independent histories; refreshing the page or
restarting the server also loses this short-term context. History is not saved to
a database, file, global agent, or shared cache.

Try this memory demonstration:

1. “What's in my portfolio?”
2. “Tell me more about my largest holding.”
3. “How many shares do I have of that stock?”

Tools now quote the share counts already stored in the synthetic CSV. In this demo,
“largest” is explicitly **by share count**: AAPL has 10 shares and MSFT has 5.
Current position market values are unavailable, so the assistant must not infer
which holding is largest by market value. If earlier context has been removed or
the referent is ambiguous, the assistant should ask for clarification.

Each portfolio request still starts with fresh synthetic data. The web demo is
read-only, and no portfolio edits can persist into another request. Conversation memory retains
what was said; current portfolio facts must be checked with tools. The UI never
selects real `holdings.csv` for portfolio requests. Cloud deployment preparation
is documented in [DEPLOYMENT.md](DEPLOYMENT.md); no public URL has been published.

## Public demo safeguards

The web adapter always selects a request-local synthetic CSV and read-only mode.
User questions and tool inputs cannot choose a CSV path. Unknown write tools are
rejected by the existing executor, and malformed position tickers return a safe
error. Questions are limited to 2,000 characters; ticker inputs accept a bounded
single symbol such as `AAPL`, `BRK-B`, or `0700.HK`.

Public research fails without displaying scores or final prose if a required
source is missing, failed, non-finite, for another ticker, or outside these demo
freshness limits:

| Evidence | Maximum age |
|---|---|
| Last price-history bar | 7 calendar days, allowing weekends/holidays |
| Fundamental data retrieval | 24 hours |
| Income, balance-sheet, and cash-flow statement periods | 550 days |
| Newest available company-news article | 7 days |

Unknown timestamps and timestamps more than five minutes in the future are
rejected. These are conservative demo policies, not trading rules. Complete core
specialist outputs, usable price indicators, company/key statistics, financial
statements, and company articles are required. Partial optional news-source
failure is noted when usable articles remain. Financial-statement dates and
retrieval times are displayed separately; a newly generated report is not a
verified live market quote. Scores remain the existing model/deterministic
assessments, and the original scoring formulas are unchanged.

Publication dates must come from the provider and carry explicit provenance
through the news cache. Missing/malformed dates are labelled unavailable rather
than replaced with the current time. Legacy news caches without that provenance
are treated as unverified by the web demo; wait for the normal company-news cache
expiry (one hour) or clear the ignored news cache locally before retrying. Every
company article used by the public report must have a usable recent date. The
first statement row consumed by the calculator must itself be usable; an older
valid row cannot conceal missing current data. Required technical score inputs
must be finite; the existing optional 200-day SMA absence remains supported.

Configured credential values, their URL-encoded forms, and common credential
error patterns are redacted from displayed reports, answers, history and tool
observations. Model text is rendered as plain text, so embedded Markdown images
cannot issue external requests. Provider exceptions are replaced with safe
messages. `.streamlit/config.toml` disables public error details and static file
serving. Store credentials in environment variables (or local ignored `.env`);
the same variable names can be injected by the deployment's secret store.

`holdings.csv` is private local data and is now ignored and removed from the
current Git snapshot; the existing local file is preserved. Prior Git history is
unchanged and may still contain that file. Do not publish the repository/history
as a sanitized public source archive without checking it; Stage 4 does not rewrite
history or change repository access. Deploy only source that excludes personal
data. Research data caches remain shared by ticker, contain public source data,
and do not store browser questions or portfolio state. Public web requests no
longer write `data/runs` records.

## Usage

**Analyze a stock** (full report + persisted record):

```bash
python lead_agent.py MU
python lead_agent.py MU --perspective "charlie-munger" --perspective-weight 0.15
python lead_agent.py MU AVGO --quick        # no final synthesis; specialists still use paid calls
```

Full research uses paid model calls and saves reports to
`data/runs/<YYYY-MM-DD>/<TICKER>.json`; a second run for the same ticker that day
overwrites its record. `--quick` prints recommendations without saving a record.
Check the generation time and pipeline notes before using
a report: default scores or missing sources are not verified market observations.

The Python equivalent is:

```python
from investment.agents.lead import GeneralAnalystAgent

agent = GeneralAnalystAgent()
print(agent.analyze("MU"))
```

**Portfolio assistant** (separate workflow, using temporary synthetic data):

```bash
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from investment.agents.portfolio import PortfolioTrackerAgent

with TemporaryDirectory() as demo_dir:
    csv = Path(demo_dir) / "synthetic.csv"
    csv.write_text(
        "status,ticker,shares,cost_basis,date_acquired,notes\n"
        "hold,AAPL,10,100,2026-01-15,Synthetic demonstration holding\n"
        "watch,MSFT,0,0,,Synthetic watchlist item\n",
        encoding="utf-8",
    )
    agent = PortfolioTrackerAgent(csv_path=str(csv))
    print(agent.query("What's in my portfolio?"))
PY
```

Portfolio queries use paid model calls. The default constructor uses the repository's
`holdings.csv`, so always pass an explicit synthetic path for demonstrations. Its
tools include `add_position` and `add_to_watchlist`, which can write to the selected
CSV. This direct Python API remains a local write-capable workflow by default;
use `read_only=True` for read-only access, as the web adapter does.
Direct `query(question)` calls remain stateless by default. To supply
context in Python, pass `conversation.format_history(history)` as the optional
`chat_history` argument, using `investment.conversation.add_exchange` to retain
bounded user/assistant exchanges. Portfolio read tools quote stored shares,
tickers, dates, and notes; they do not calculate current position market values.

**Scheduled evaluation** (headless; this is what cron/launchd should invoke):

```bash
python runner.py --dry-run                  # preview universe + config, free
python runner.py                            # evaluate universe from holdings.csv / runner.toml
python runner.py --tickers MU AVGO          # explicit universe
```

`--dry-run` avoids analysis/model calls but still creates a run log. Other runner
commands use paid model calls and can read the existing holdings universe. For a
setup check that does not read holdings, use `python runner.py --help`.

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

**Manage the universe** — the local ignored `holdings.csv` tracks *which* tickers
are held or watched (status, date, notes). It is not a P&L tracker. Fresh clones do
not include personal holdings; the web demo generates its synthetic CSV itself.

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
tests/                 offline pytest suite, including agent startup checks
scripts/               live-API smoke scripts (spend tokens; run deliberately)
lead_agent.py, runner.py, distill_investor.py, X_fetch.py
                       root entrypoint shims for the commands above
```

## Testing

```bash
python -m pip install -e ".[web,dev]"
python -m pip check
python -m pytest       # offline unit suite; no API keys, no network
python -m pytest tests/test_startup.py
python -m pytest tests/test_web.py tests/test_streamlit_app.py
python lead_agent.py --help
python runner.py --help
```

Startup tests create real model clients with placeholder credentials and temporary
cache/portfolio paths, and block network connections. The portfolio tool test uses
a fake model and synthetic CSV. The test session redirects third-party caches to a
temporary directory before collection, so dependency imports cannot migrate your
existing Yahoo cache. No tests contact Anthropic or inspect real holdings.
The `scripts/test_*.py` files are live smoke scripts that need credentials and can
spend tokens; they are excluded from the normal pytest suite.

## Troubleshooting

- `ANTHROPIC_API_KEY environment variable not set`: set a non-empty key in the
  repository-root `.env` or your shell before initializing an agent. An invalid key
  or unavailable model is detected only when a live request is made.
- Import errors: activate the environment used for installation, check
  `python --version` and `python -m pip --version`, then install from the repo root.
- Dependency conflicts: use a fresh Python 3.11 environment and run
  `python -m pip check`. Keep the legacy LangChain pins: the portfolio executor uses
  `AgentExecutor` and `create_react_agent`. Do not upgrade LangChain independently.
  The older local `invest` environment had an unrelated newer
  `langchain-text-splitters` installation incompatible with `langchain-core==0.1.23`;
  do not copy that environment wholesale.
- PyPI download errors: check network/index access. A package-download failure does
  not establish a conflict in the project's dependency constraints.
- `yfinance-cache` Manager-lock warnings: restricted runtimes may block its local
  multiprocessing socket. The library falls back to thread locks; recheck data
  retrieval in the intended runtime before a live demo.
- API timeouts, rate limits, or empty sources: check service access and quotas;
  inspect report pipeline notes. Missing data must not be treated as a verified
  observation. Do not publish this local workflow with real holdings or secrets.
