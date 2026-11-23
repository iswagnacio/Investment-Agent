"""
Universe Manager
================
Phase 1 of ROADMAP.md: this module manages the ticker universe the analysis
pipeline evaluates — which tickers are held vs. watched — not P&L. The CSV
keeps its historical columns (shares, cost_basis) so existing files load, but
nothing here computes cost or profit anymore; evaluation quality is measured
by signal track record (data/runs/), not returns.
"""

import pandas as pd

from investment import config
from pathlib import Path
from typing import List


class PortfolioManager:
    """Manages the ticker universe stored in holdings.csv."""

    def __init__(self, csv_path: str = str(config.HOLDINGS_CSV), *, read_only: bool = False):
        self.csv_path = Path(csv_path)
        self.read_only = read_only
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create portfolio file if it doesn't exist"""
        if not self.csv_path.exists():
            if self.read_only:
                raise ValueError("Read-only portfolio data is unavailable.")
            df = pd.DataFrame(columns=[
                'status', 'ticker', 'shares', 'cost_basis',
                'date_acquired', 'notes'
            ])
            df.to_csv(self.csv_path, index=False)

    def read_portfolio(self) -> pd.DataFrame:
        """Read portfolio from CSV"""
        return pd.read_csv(self.csv_path)

    def get_universe(self) -> List[str]:
        """All tickers to evaluate — holdings first, then watchlist.
        This is what a scheduled run (Phase 2 runner) iterates over."""
        df = self.read_portfolio()
        held = df[df['status'] == 'hold']['ticker'].str.upper().tolist()
        watched = df[df['status'] == 'watch']['ticker'].str.upper().tolist()
        seen = set()
        return [t for t in held + watched if not (t in seen or seen.add(t))]

    def get_holdings(self) -> str:
        """List positions with 'hold' status (tickers to evaluate, not P&L)."""
        df = self.read_portfolio()
        holdings = df[df['status'] == 'hold']

        if holdings.empty:
            return "No current holdings in portfolio."

        result = "Current Holdings:\n\n"
        for _, row in holdings.iterrows():
            result += f"• {row['ticker']} (since {row['date_acquired']})"
            shares = row['shares'] if pd.notna(row['shares']) else 'Unavailable'
            result += f" — Shares: {shares}"
            if pd.notna(row['notes']) and row['notes']:
                result += f" — {row['notes']}"
            result += "\n"
        return result

    def get_watchlist(self) -> str:
        """Get all positions with 'watch' status"""
        df = self.read_portfolio()
        watchlist = df[df['status'] == 'watch']

        if watchlist.empty:
            return "Watchlist is empty."

        result = "Watchlist:\n\n"
        for _, row in watchlist.iterrows():
            result += f"• {row['ticker']}"
            if pd.notna(row['notes']) and row['notes']:
                result += f" - {row['notes']}"
            result += "\n"

        return result

    def get_position_details(self, ticker: str) -> str:
        """Get details for a specific ticker"""
        df = self.read_portfolio()
        position = df[df['ticker'].str.upper() == ticker.upper()]

        if position.empty:
            return f"No position found for {ticker}."

        row = position.iloc[0]
        shares = row['shares'] if pd.notna(row['shares']) else 'Unavailable'
        quantity = f"Shares: {shares}\n" if row['status'] == 'hold' else ""
        return (f"{'Position' if row['status'] == 'hold' else 'Watchlist item'}: {row['ticker']}\n"
                f"Status: {row['status']}\n"
                f"{quantity}"
                f"Date Acquired: {row['date_acquired'] if pd.notna(row['date_acquired']) else '—'}\n"
                f"Notes: {row['notes'] if pd.notna(row['notes']) else 'None'}")

    def add_position(self, ticker: str, shares: float, cost_basis: float,
                    date_acquired: str, notes: str = "") -> str:
        """Record a held ticker. shares/cost_basis are stored for the CSV's
        historical shape but are not used in any calculation."""
        if self.read_only:
            raise PermissionError("This portfolio is read-only.")
        df = self.read_portfolio()

        if ticker.upper() in df['ticker'].str.upper().values:
            return f"Error: {ticker} already exists in portfolio. Use update function instead."

        new_row = pd.DataFrame([{
            'status': 'hold',
            'ticker': ticker.upper(),
            'shares': shares,
            'cost_basis': cost_basis,
            'date_acquired': date_acquired,
            'notes': notes
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

        return f"Successfully added {ticker.upper()} to holdings."

    def add_to_watchlist(self, ticker: str, notes: str = "") -> str:
        """Add a ticker to watchlist"""
        if self.read_only:
            raise PermissionError("This portfolio is read-only.")
        df = self.read_portfolio()

        if ticker.upper() in df['ticker'].str.upper().values:
            return f"Error: {ticker} already exists in portfolio or watchlist."

        new_row = pd.DataFrame([{
            'status': 'watch',
            'ticker': ticker.upper(),
            'shares': 0,
            'cost_basis': 0,
            'date_acquired': '',
            'notes': notes
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

        return f"Successfully added {ticker} to watchlist."

    def get_portfolio_summary(self) -> str:
        """High-level universe summary (counts + tickers; no cost/P&L math)."""
        df = self.read_portfolio()

        holdings = df[df['status'] == 'hold']
        watchlist = df[df['status'] == 'watch']

        summary = "Universe Summary:\n\n"
        summary += f"Holdings:  {len(holdings)} ({', '.join(holdings['ticker']) or 'none'})\n"
        summary += f"Watchlist: {len(watchlist)} ({', '.join(watchlist['ticker']) or 'none'})\n"
        return summary
