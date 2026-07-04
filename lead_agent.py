"""Compatibility shim — code moved to investment/agents/lead.py (Phase 4).

Preserves both the CLI (python lead_agent.py MU --perspective ...) and legacy
imports (from lead_agent import GeneralAnalystAgent). New code should import
from investment.agents.lead / investment.scoring directly.
"""

from investment.agents.lead import *          # noqa: F401,F403
from investment.agents.lead import main
from investment.scoring import ScoreCalculator, Signal  # noqa: F401

if __name__ == "__main__":
    main()
