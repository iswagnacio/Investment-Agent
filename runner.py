"""Entrypoint shim — implementation lives in investment/runner.py (Phase 4).

Keeps the documented scheduler command working:
    python runner.py [--config ...] [--tickers ...] [--dry-run]
"""

from investment.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
