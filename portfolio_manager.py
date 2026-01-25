import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PortfolioManager:
    """Handles portfolio data operations"""
    
    def __init__(self, csv_path: str = "holdings.csv"):
        self.csv_path = Path(csv_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create portfolio file if it doesn't exist"""
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=[
                'status', 'ticker', 'shares', 'cost_basis', 
                'date_acquired', 'notes'
            ])
            df.to_csv(self.csv_path, index=False)
    
    def read_portfolio(self) -> pd.DataFrame:
        """Read portfolio from CSV"""
        return pd.read_csv(self.csv_path)
    
    def get_holdings(self) -> str:
        """Get all positions with 'hold' status"""
        df = self.read_portfolio()
        holdings = df[df['status'] == 'hold'].copy()
        
        if holdings.empty:
            return "No current holdings in portfolio."
        
        # Calculate total value and allocation (placeholder - needs real prices)
        holdings['total_cost'] = holdings['shares'] * holdings['cost_basis']
        total_portfolio_cost = holdings['total_cost'].sum()
        holdings['allocation_pct'] = (holdings['total_cost'] / total_portfolio_cost * 100).round(2)
        
        result = "Current Holdings:\n\n"
        for _, row in holdings.iterrows():
            result += f"• {row['ticker']}: {row['shares']} shares @ ${row['cost_basis']:.2f}\n"
            result += f"  Cost Basis: ${row['total_cost']:.2f} ({row['allocation_pct']:.1f}% of portfolio)\n"
            result += f"  Acquired: {row['date_acquired']}\n"
            if pd.notna(row['notes']):
                result += f"  Notes: {row['notes']}\n"
            result += "\n"
        
        result += f"Total Portfolio Cost Basis: ${total_portfolio_cost:,.2f}"
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
            if pd.notna(row['notes']):
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
        
        if row['status'] == 'hold':
            total_cost = row['shares'] * row['cost_basis']
            return (f"Position Details for {row['ticker']}:\n"
                   f"Status: {row['status']}\n"
                   f"Shares: {row['shares']}\n"
                   f"Cost Basis: ${row['cost_basis']:.2f}\n"
                   f"Total Cost: ${total_cost:.2f}\n"
                   f"Date Acquired: {row['date_acquired']}\n"
                   f"Notes: {row['notes'] if pd.notna(row['notes']) else 'None'}")
        else:
            return (f"Watchlist Item: {row['ticker']}\n"
                   f"Notes: {row['notes'] if pd.notna(row['notes']) else 'None'}")
    
    def add_position(self, ticker: str, shares: float, cost_basis: float, 
                    date_acquired: str, notes: str = "") -> str:
        """Add a new holding to portfolio"""
        df = self.read_portfolio()
        
        # Check if ticker already exists
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
        
        return f"Successfully added {shares} shares of {ticker} at ${cost_basis:.2f} per share."
    
    def add_to_watchlist(self, ticker: str, notes: str = "") -> str:
        """Add a ticker to watchlist"""
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
        """Get high-level portfolio summary"""
        df = self.read_portfolio()
        
        holdings = df[df['status'] == 'hold']
        watchlist = df[df['status'] == 'watch']
        
        summary = "Portfolio Summary:\n\n"
        summary += f"Total Holdings: {len(holdings)} positions\n"
        summary += f"Watchlist Items: {len(watchlist)} tickers\n"
        
        if not holdings.empty:
            total_cost = (holdings['shares'] * holdings['cost_basis']).sum()
            summary += f"Total Cost Basis: ${total_cost:,.2f}\n"
            summary += f"\nTop Holdings by Cost Basis:\n"
            
            holdings_copy = holdings.copy()
            holdings_copy['total_cost'] = holdings_copy['shares'] * holdings_copy['cost_basis']
            top_3 = holdings_copy.nlargest(3, 'total_cost')
            
            for _, row in top_3.iterrows():
                pct = (row['total_cost'] / total_cost * 100)
                summary += f"  • {row['ticker']}: ${row['total_cost']:,.2f} ({pct:.1f}%)\n"
        
        return summary