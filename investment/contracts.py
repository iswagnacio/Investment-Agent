"""
Shared Contracts
================
Phase 1 of ROADMAP.md: the structured, persisted result of one evaluation.

An AnalysisRecord is built FIRST, saved to data/runs/<YYYY-MM-DD>/<TICKER>.json,
and only then rendered into a human report (reporting.render_report). This is
the same structure-then-render discipline the distiller already uses
(framework.json -> SKILL.md): the JSON is the product, prose is a view of it.

The record deliberately carries qualitative judgment only — thesis, signal,
confidence, risks — no price targets, options strategies, or P&L. Scheduled
runs (Phase 2) diff consecutive records per ticker; self-created assessments
(Phase 3) earn blend weight from the track record these files accumulate.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from investment import config

SCHEMA_VERSION = 1


@dataclass
class AnalysisRecord:
    """The persisted result of one stock evaluation."""

    # Identity
    ticker: str
    company: str = ""
    sector: str = ""
    industry: str = ""
    current_price: Optional[float] = None

    # Provenance
    generated_at: str = ""          # ISO timestamp; set by __post_init__ if empty
    run_id: str = ""                # config.get_run_id() of the producing process
    model: str = ""                 # lead model ID used for synthesis
    schema_version: int = SCHEMA_VERSION

    # Per-domain signals: {"technical": {"score": ..., "structured": {...}}, ...}
    # Perspective (when applied) appears under "perspective" with its verdict,
    # narrative, and distillation metadata.
    signals: Dict[str, Any] = field(default_factory=dict)

    # Blend actually used (sums to 1.0) and the outcome
    weights: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 50.0
    signal: str = "Hold"
    confidence: float = 0.0

    # Qualitative judgment
    thesis: str = ""                # full LLM synthesis text
    key_risks: List[str] = field(default_factory=list)

    # Pipeline notes: detail fetches, data errors, degradations
    data_flags: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.ticker = self.ticker.upper()
        if not self.generated_at:
            self.generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        if not self.run_id:
            self.run_id = config.get_run_id()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, runs_dir: Optional[Path] = None) -> Path:
        """Write to <runs_dir>/<YYYY-MM-DD>/<TICKER>.json (same-day runs overwrite).

        runs_dir defaults to config.RUNS_DIR at CALL time (not import time),
        so tests and runtime overrides can redirect it.
        """
        day = self.generated_at[:10] or f"{datetime.now():%Y-%m-%d}"
        out_dir = Path(runs_dir or config.RUNS_DIR) / day
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.ticker}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        return path

    @classmethod
    def load(cls, path: Path) -> "AnalysisRecord":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def latest(cls, ticker: str,
               runs_dir: Optional[Path] = None,
               before_day: Optional[str] = None) -> Optional["AnalysisRecord"]:
        """Most recent saved record for `ticker`; `before_day` ('YYYY-MM-DD',
        exclusive) lets Phase 2 diff today's run against the previous one."""
        ticker = ticker.upper()
        root = Path(runs_dir or config.RUNS_DIR)
        if not root.is_dir():
            return None
        days = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
        for day_dir in days:
            if before_day and day_dir.name >= before_day:
                continue
            path = day_dir / f"{ticker}.json"
            if path.is_file():
                return cls.load(path)
        return None
