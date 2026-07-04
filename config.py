"""
Central Configuration
=====================
Single source of truth for paths, model names, blend weights, and the small
shared helpers that were previously duplicated across modules (Phase 0 of
ROADMAP.md).

Paths are anchored to the repo root via this file's location, so every
entrypoint works regardless of the current working directory — a requirement
for scheduled (cron/launchd) runs.

Usage:
    import config
    cache = config.PERSPECTIVES_CACHE_DIR      # pathlib.Path
    key = config.normalize_key("Charlie Munger")
    log = config.get_logger(__name__)
"""

import logging
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# =============================================================================
# Paths (all anchored to the repo root, never the cwd)
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FUNDAMENTAL_CACHE_DIR = DATA_DIR / "fundamental_cache"
NEWS_CACHE_DIR = DATA_DIR / "news_cache"
PERSPECTIVES_CACHE_DIR = DATA_DIR / "perspectives_cache"
X_CACHE_DIR = DATA_DIR / "x_cache"

RUNS_DIR = DATA_DIR / "runs"              # Phase 1: persisted AnalysisRecords
WORKSPACES_DIR = DATA_DIR / "workspaces"  # Phase 3: per-agent environments

HOLDINGS_CSV = ROOT / "holdings.csv"

# =============================================================================
# Models
# =============================================================================
# STRONG_MODEL: synthesis, extraction, per-stock analysis.
# CHEAP_MODEL: cleanup passes, tool-calling chores.

STRONG_MODEL = "claude-sonnet-4-5"
CHEAP_MODEL = "claude-haiku-4-5"

# =============================================================================
# Scoring
# =============================================================================
# Base blend across the three always-on domains. Must sum to 1.0.
# calculate_scores() post-blends a perspective on top by rescaling these by
# (1 - perspective_weight); see lead_agent.calculate_scores.

BLEND_WEIGHTS = {
    "technical": 0.35,
    "fundamental": 0.45,
    "sentiment": 0.20,
}

# Perspective influence is capped and then shrunk further when the
# distillation confidence is Medium/Low. Raise only after backtest evidence.
MAX_PERSPECTIVE_WEIGHT = 0.25
DEFAULT_PERSPECTIVE_WEIGHT = 0.15

# =============================================================================
# Shared helpers (previously duplicated per-module)
# =============================================================================

def normalize_key(name: str) -> str:
    """'Serenity@aleabitoreddit' -> 'serenity-aleabitoreddit'; 'Charlie Munger' -> 'charlie-munger'.

    The one canonical implementation: the distiller uses it to choose the cache
    directory and the perspective agent uses it to find that directory again.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence from an LLM reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


# =============================================================================
# Logging
# =============================================================================

_RUN_ID: str = ""


def get_run_id() -> str:
    """Stable ID for this process, e.g. '20260703-142705-3f2a'. Lazy-created."""
    global _RUN_ID
    if not _RUN_ID:
        _RUN_ID = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    return _RUN_ID


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> str:
    """Configure root logging once per process; returns the run ID.

    Call from entrypoints (CLIs, the future runner). Library modules should
    just use get_logger(__name__) and never configure handlers themselves.
    """
    run_id = get_run_id()
    fmt = logging.Formatter(
        f"%(asctime)s [{run_id}] %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        root.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return run_id


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
