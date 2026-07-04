"""Entrypoint shim — implementation lives in investment/offline/distill_investor.py.

    python distill_investor.py "Charlie Munger" [--workspace] [--from-brief] ...
"""

from investment.offline.distill_investor import main

if __name__ == "__main__":
    main()
