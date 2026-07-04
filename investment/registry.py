"""
Assessment Registry
===================
Phase 3 of ROADMAP.md: agents become registry entries, not code edits.

registry.toml declares:
  [core]           — blend weights for the three always-on domains
                     (replaces hardcoded config.BLEND_WEIGHTS when present)
  [agents.<key>]   — custom assessments. Each is backed by a self-authored
                     methodology artifact at data/workspaces/<key>/framework.json
                     (the same framework.json + SKILL.md + metadata.json shape
                     the perspective distiller produces), applied at runtime by
                     the same machinery that applies investor perspectives.

Trust is earned, never declared:
  effective weight = min(weight, max_weight)
                     x artifact-confidence factor   (High 1.0 / Medium 0.7 / Low 0.4)
                     x maturity ramp                (25% at first run -> 100% by MATURITY_RUNS)
and the combined influence of ALL post-blend assessments (perspective included)
is capped at MAX_TOTAL_POST_WEIGHT so the core domains always keep the majority.

Permissions are enforced where the data flows: an agent only receives the data
sources its manifest lists, and its per-run LLM budget must cover the applier's
single call or it is skipped.
"""

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from investment import config

log = config.get_logger("registry")

DEFAULT_REGISTRY_PATH = config.ROOT / "registry.toml"

# A new agent's influence ramps from 25% of its weight to 100% as it
# accumulates this many persisted appearances in data/runs/.
MATURITY_RUNS = 20
MATURITY_FLOOR = 0.25

# Post-blended assessments (perspective + customs) can never outweigh the core.
MAX_TOTAL_POST_WEIGHT = 0.40

_CONFIDENCE_FACTOR = {"High": 1.0, "Medium": 0.7, "Low": 0.4}

VALID_DATA_SOURCES = ("technical", "fundamental", "sentiment")


@dataclass
class Permissions:
    """What a custom agent may see and spend. Enforced at the data boundary."""
    data_sources: List[str] = field(default_factory=lambda: list(VALID_DATA_SOURCES))
    max_llm_calls: int = 1
    workspace_write: bool = False   # authoring-time permission (offline tools)


@dataclass
class CustomAgentSpec:
    key: str
    enabled: bool = True
    weight: float = 0.10
    max_weight: float = 0.10
    permissions: Permissions = field(default_factory=Permissions)

    @property
    def workspace_dir(self) -> Path:
        return config.WORKSPACES_DIR / self.key

    def artifact_confidence(self) -> str:
        """Read the artifact's own quality gate (Low if absent — fail closed)."""
        meta_path = self.workspace_dir / "metadata.json"
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            return meta.get("quality_metrics", {}).get("confidence", "Low")
        except Exception:
            return "Low"


@dataclass
class Registry:
    core_weights: Dict[str, float] = field(
        default_factory=lambda: dict(config.BLEND_WEIGHTS))
    custom: Dict[str, CustomAgentSpec] = field(default_factory=dict)

    def enabled_custom(self) -> List[CustomAgentSpec]:
        return [s for s in self.custom.values() if s.enabled]


def load_registry(path: Optional[Path] = None) -> Registry:
    """Read registry.toml; missing file -> defaults (config weights, no customs)."""
    path = Path(path or DEFAULT_REGISTRY_PATH)
    reg = Registry()
    if not path.is_file():
        return reg

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    core = raw.get("core", {})
    if core:
        weights = {k: float(core[k]) for k in ("technical", "fundamental", "sentiment")
                   if k in core}
        if set(weights) != set(VALID_DATA_SOURCES):
            raise ValueError(f"[core] must define all of {VALID_DATA_SOURCES}, got {list(weights)}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"[core] weights must sum to 1.0 (got {total})")
        reg.core_weights = weights

    for key, spec_raw in (raw.get("agents", {}) or {}).items():
        perms_raw = spec_raw.get("permissions", {})
        sources = perms_raw.get("data_sources", list(VALID_DATA_SOURCES))
        bad = [s for s in sources if s not in VALID_DATA_SOURCES]
        if bad:
            raise ValueError(f"agents.{key}: unknown data_sources {bad}")
        spec = CustomAgentSpec(
            key=key,
            enabled=bool(spec_raw.get("enabled", True)),
            weight=float(spec_raw.get("weight", 0.10)),
            max_weight=float(spec_raw.get("max_weight", 0.10)),
            permissions=Permissions(
                data_sources=sources,
                max_llm_calls=int(perms_raw.get("max_llm_calls", 1)),
                workspace_write=bool(perms_raw.get("workspace_write", False)),
            ),
        )
        reg.custom[key] = spec
    return reg


# =============================================================================
# Trust-gated weighting
# =============================================================================

def appearances(agent_key: str, runs_dir: Optional[Path] = None) -> int:
    """How many persisted AnalysisRecords carry this agent's signal — the
    maturity input. Track-record *correctness* scoring builds on this later."""
    root = Path(runs_dir or config.RUNS_DIR)
    if not root.is_dir():
        return 0
    count = 0
    for path in root.glob("*/*.json"):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            if agent_key in raw.get("signals", {}):
                count += 1
        except Exception:
            continue
    return count


def effective_custom_weight(spec: CustomAgentSpec,
                            artifact_confidence: Optional[str] = None,
                            n_appearances: Optional[int] = None,
                            runs_dir: Optional[Path] = None) -> float:
    """Requested weight, capped, then shrunk by artifact quality and maturity."""
    base = min(spec.weight, spec.max_weight, config.MAX_PERSPECTIVE_WEIGHT)
    base = max(0.0, base)
    conf = artifact_confidence if artifact_confidence is not None \
        else spec.artifact_confidence()
    conf_factor = _CONFIDENCE_FACTOR.get(conf, 0.4)
    n = n_appearances if n_appearances is not None \
        else appearances(spec.key, runs_dir)
    maturity = max(MATURITY_FLOOR, min(1.0, n / MATURITY_RUNS))
    return round(base * conf_factor * maturity, 3)


def cap_post_blend(entries: List[tuple]) -> List[tuple]:
    """entries: [(name, score, weight)]. If the summed weight exceeds
    MAX_TOTAL_POST_WEIGHT, scale all weights down proportionally so the core
    domains keep the majority no matter how many assessments are registered."""
    total = sum(w for _, _, w in entries)
    if total <= MAX_TOTAL_POST_WEIGHT or total == 0:
        return entries
    factor = MAX_TOTAL_POST_WEIGHT / total
    return [(name, score, round(w * factor, 3)) for name, score, w in entries]


# =============================================================================
# Workspaces
# =============================================================================

def ensure_workspace(key: str) -> Path:
    """Create (if needed) and return an agent's workspace directory — its
    environment for self-authored artifacts, notes, and caches."""
    ws = config.WORKSPACES_DIR / config.normalize_key(key)
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def list_workspaces() -> List[str]:
    """Workspace keys that contain a usable methodology artifact."""
    root = config.WORKSPACES_DIR
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if (d / "framework.json").is_file())
