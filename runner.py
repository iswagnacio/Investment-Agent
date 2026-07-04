"""
Headless Evaluation Runner
==========================
Phase 2 of ROADMAP.md: the entrypoint a scheduler invokes. The schedule itself
lives OUTSIDE this repo (cron / launchd / a Claude Code routine); this module's
job is to be safely invokable: cwd-independent, logged, idempotent per day,
and honest with its exit code.

    python runner.py                       # universe from holdings.csv, runner.toml if present
    python runner.py --tickers MU AVGO     # explicit universe
    python runner.py --config my.toml      # alternate config
    python runner.py --dry-run             # resolve universe + config, run nothing

Per run it writes, under data/runs/<YYYY-MM-DD>/:
    <TICKER>.json   one AnalysisRecord per ticker (via lead_agent)
    digest.md       what CHANGED vs. each ticker's previous record — signal
                    flips, notable score moves, new tickers, failures. A
                    schedule that re-prints the same report daily is noise;
                    the diff is the product.
    run.log         full log of the run

Exit codes: 0 = all tickers evaluated; 1 = some failed; 2 = nothing ran
(empty universe / bad config).
"""

import argparse
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from contracts import AnalysisRecord

log = config.get_logger("runner")

DEFAULT_CONFIG_PATH = config.ROOT / "runner.toml"

# Score move (in points) worth reporting even without a signal flip.
NOTABLE_SCORE_DELTA = 10.0


# =============================================================================
# Config + universe
# =============================================================================

def load_runner_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read runner.toml; missing file -> defaults (holdings universe, no perspective)."""
    cfg: Dict[str, Any] = {
        "tickers": [],                # empty -> holdings.csv universe
        "perspective": None,
        "perspective_weight": config.DEFAULT_PERSPECTIVE_WEIGHT,
        "use_specialist_agents": True,
        "parallel": 1,
    }
    path = path or DEFAULT_CONFIG_PATH
    if not Path(path).is_file():
        return cfg
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    universe = raw.get("universe", {})
    analysis = raw.get("analysis", {})
    run = raw.get("run", {})
    cfg["tickers"] = [t.upper() for t in universe.get("tickers", [])]
    cfg["perspective"] = analysis.get("perspective") or None
    cfg["perspective_weight"] = float(
        analysis.get("perspective_weight", cfg["perspective_weight"]))
    cfg["use_specialist_agents"] = bool(
        analysis.get("use_specialist_agents", True))
    cfg["parallel"] = max(1, int(run.get("parallel", 1)))
    return cfg


def resolve_universe(cfg: Dict[str, Any], cli_tickers: Optional[List[str]] = None) -> List[str]:
    """CLI tickers beat config tickers beat holdings.csv."""
    if cli_tickers:
        return [t.upper() for t in cli_tickers]
    if cfg["tickers"]:
        return cfg["tickers"]
    from portfolio import PortfolioManager  # deferred: pandas import
    return PortfolioManager().get_universe()


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class TickerOutcome:
    ticker: str
    record: Optional[AnalysisRecord] = None
    previous: Optional[AnalysisRecord] = None
    error: str = ""


def evaluate_ticker(agent, ticker: str, cfg: Dict[str, Any], today: str) -> TickerOutcome:
    """Evaluate one ticker; never raises — failures land in .error."""
    outcome = TickerOutcome(ticker=ticker)
    try:
        outcome.previous = AnalysisRecord.latest(ticker, before_day=today)
    except Exception as e:  # a corrupt old record must not block today's run
        log.warning("%s: could not load previous record: %s", ticker, e)
    try:
        outcome.record = agent.analyze_to_record(
            ticker,
            perspective=cfg["perspective"],
            perspective_weight=cfg["perspective_weight"],
        )
        log.info("%s: %s (score %.1f, confidence %.0f%%)", ticker,
                 outcome.record.signal, outcome.record.overall_score,
                 outcome.record.confidence)
    except Exception as e:
        outcome.error = str(e)
        log.error("%s: evaluation failed: %s", ticker, e)
    return outcome


def _make_agent(cfg: Dict[str, Any]):
    from lead_agent import GeneralAnalystAgent  # deferred: heavy imports
    return GeneralAnalystAgent(
        use_specialist_agents=cfg["use_specialist_agents"],
        use_perspective=bool(cfg["perspective"]),
    )


def run_universe(universe: List[str], cfg: Dict[str, Any], today: str) -> List[TickerOutcome]:
    if cfg["parallel"] > 1:
        # One agent per worker: agents hold LLM clients and per-ticker state.
        outcomes: List[TickerOutcome] = []
        with ThreadPoolExecutor(max_workers=cfg["parallel"]) as pool:
            futures = {
                pool.submit(evaluate_ticker, _make_agent(cfg), t, cfg, today): t
                for t in universe
            }
            for fut in as_completed(futures):
                outcomes.append(fut.result())
        order = {t: i for i, t in enumerate(universe)}
        return sorted(outcomes, key=lambda o: order[o.ticker])
    agent = _make_agent(cfg)
    return [evaluate_ticker(agent, t, cfg, today) for t in universe]


# =============================================================================
# Digest: only what changed
# =============================================================================

def classify_outcome(o: TickerOutcome) -> str:
    """-> 'failed' | 'new' | 'signal_change' | 'notable_move' | 'unchanged'"""
    if o.error or not o.record:
        return "failed"
    if o.previous is None:
        return "new"
    if o.record.signal != o.previous.signal:
        return "signal_change"
    if abs(o.record.overall_score - o.previous.overall_score) >= NOTABLE_SCORE_DELTA:
        return "notable_move"
    return "unchanged"


def render_digest(outcomes: List[TickerOutcome], run_id: str, day: str) -> str:
    groups: Dict[str, List[TickerOutcome]] = {
        "signal_change": [], "notable_move": [], "new": [], "failed": [], "unchanged": [],
    }
    for o in outcomes:
        groups[classify_outcome(o)].append(o)

    ok = len(outcomes) - len(groups["failed"])
    lines = [
        f"# Run digest — {day}",
        "",
        f"Run `{run_id}`: {ok}/{len(outcomes)} tickers evaluated.",
        "",
    ]

    if groups["signal_change"]:
        lines.append("## Signal changes")
        for o in groups["signal_change"]:
            p, r = o.previous, o.record
            lines.append(
                f"- **{o.ticker}: {p.signal} → {r.signal}** "
                f"(score {p.overall_score:.1f} → {r.overall_score:.1f}, "
                f"confidence {p.confidence:.0f}% → {r.confidence:.0f}%)")
        lines.append("")

    if groups["notable_move"]:
        lines.append(f"## Notable score moves (±{NOTABLE_SCORE_DELTA:.0f}+, signal unchanged)")
        for o in groups["notable_move"]:
            p, r = o.previous, o.record
            lines.append(f"- {o.ticker}: {p.overall_score:.1f} → {r.overall_score:.1f} ({r.signal})")
        lines.append("")

    if groups["new"]:
        lines.append("## New (no previous record)")
        for o in groups["new"]:
            r = o.record
            lines.append(f"- {o.ticker}: {r.signal} (score {r.overall_score:.1f})")
        lines.append("")

    if groups["failed"]:
        lines.append("## Failures")
        for o in groups["failed"]:
            lines.append(f"- {o.ticker}: {o.error or 'no record produced'}")
        lines.append("")

    if not (groups["signal_change"] or groups["notable_move"] or groups["new"] or groups["failed"]):
        lines += ["No signal changes, notable moves, or failures.", ""]

    if groups["unchanged"]:
        lines.append("## Unchanged")
        lines.append(", ".join(
            f"{o.ticker} ({o.record.signal} {o.record.overall_score:.1f})"
            for o in groups["unchanged"]))
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Entrypoint
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Headless scheduled evaluation run.")
    ap.add_argument("--config", type=Path, default=None,
                    help=f"Runner config (default: {DEFAULT_CONFIG_PATH.name} if present)")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Explicit universe (overrides config and holdings.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve config + universe, run nothing")
    args = ap.parse_args(argv)

    today = f"{datetime.now():%Y-%m-%d}"
    day_dir = config.RUNS_DIR / today
    run_id = config.setup_logging(log_file=day_dir / "run.log")

    cfg = load_runner_config(args.config)
    universe = resolve_universe(cfg, args.tickers)
    if not universe:
        log.error("Universe is empty: no --tickers, no config tickers, no holdings.")
        return 2

    log.info("Run %s: %d tickers (%s); perspective=%s parallel=%d",
             run_id, len(universe), ", ".join(universe),
             cfg["perspective"] or "none", cfg["parallel"])

    if args.dry_run:
        print(f"run_id:      {run_id}")
        print(f"universe:    {', '.join(universe)}")
        print(f"perspective: {cfg['perspective'] or 'none'} "
              f"(weight {cfg['perspective_weight']})")
        print(f"parallel:    {cfg['parallel']}")
        print(f"output:      {day_dir}/")
        return 0

    outcomes = run_universe(universe, cfg, today)

    digest = render_digest(outcomes, run_id, today)
    day_dir.mkdir(parents=True, exist_ok=True)
    digest_path = day_dir / "digest.md"
    digest_path.write_text(digest, encoding="utf-8")
    log.info("Digest -> %s", digest_path)
    print(digest)

    failed = [o for o in outcomes if o.error or not o.record]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
