"""Compatibility shim — code moved to investment/offline/x_fetch.py (Phase 4).

    python X_fetch.py <handle> [--cache-dir ...]
"""

from investment.offline.x_fetch import *      # noqa: F401,F403
from investment.offline.x_fetch import _main

if __name__ == "__main__":
    _main()
