"""Resolve repository imports and isolate third-party caches before collection."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_sessionstart(session):
    # yfinance-cache may migrate its cache at import time, before fixtures run.
    # Older platformdirs versions on macOS do not honor XDG_CACHE_HOME.
    cache = TemporaryDirectory(prefix="investment-test-cache-")
    session.config.add_cleanup(cache.cleanup)
    cache_path = patch("platformdirs.user_cache_dir", return_value=cache.name)
    cache_path.start()
    session.config.add_cleanup(cache_path.stop)
