"""
Fundamental Data Fetcher Module
================================
Fetches financial data from multiple sources and calculates comprehensive
fundamental indicators for investment analysis.

Data Sources:
- Yahoo Finance (yfinance) - Financial statements, ratios, company info
- SEC EDGAR (via sec-api or requests) - Official filings (10-K, 10-Q, 8-K)

Calculated Metrics:
- Valuation ratios (P/E, P/B, P/S, EV/EBITDA, PEG)
- Profitability metrics (ROE, ROA, ROIC, margins)
- Liquidity ratios (Current, Quick, Cash)
- Leverage ratios (Debt/Equity, Interest Coverage)
- Efficiency metrics (Asset Turnover, Inventory Turnover)
- Growth metrics (Revenue, Earnings, Book Value growth)
- Dividend metrics (Yield, Payout Ratio, Coverage)
- Quality scores (Altman Z-Score, Piotroski F-Score, DuPont Analysis)

Usage:
    from fundamental import FundamentalCalculator
    
    calc = FundamentalCalculator()
    data = calc.get_all_fundamentals("AAPL")
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FinancialStatement:
    """Standardized financial statement data"""
    period: str  # 'annual' or 'quarterly'
    date: str
    data: Dict[str, float]
    
    def get(self, key: str, default: float = 0.0) -> float:
        return self.data.get(key, default)


@dataclass
class CompanyInfo:
    """Company metadata"""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    employees: int
    description: str
    website: str
    country: str
    exchange: str
    currency: str


@dataclass
class FundamentalData:
    """Complete fundamental data container"""
    ticker: str
    company_info: CompanyInfo
    income_statement: List[FinancialStatement]
    balance_sheet: List[FinancialStatement]
    cash_flow: List[FinancialStatement]
    valuation_ratios: Dict[str, float]
    profitability_ratios: Dict[str, float]
    liquidity_ratios: Dict[str, float]
    leverage_ratios: Dict[str, float]
    efficiency_ratios: Dict[str, float]
    growth_metrics: Dict[str, float]
    dividend_metrics: Dict[str, Any]
    quality_scores: Dict[str, Any]
    peer_comparison: Dict[str, Any]
    historical_comparison: Dict[str, Any]
    fetch_time: datetime = field(default_factory=datetime.now)


# =============================================================================
# Cache Manager
# =============================================================================

class FundamentalCache:
    """Simple file-based cache for fundamental data"""
    
    def __init__(self, cache_dir: str = "data/fundamental_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker.upper()}_fundamental.json")
    
    def get(self, ticker: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Get cached data if fresh enough"""
        cache_path = self._get_cache_path(ticker)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_at > timedelta(hours=max_age_hours):
                return None
            logger.info(f"Cache hit for {ticker}")
            return data
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, ticker: str, data: Dict):
        """Cache data"""
        cache_path = self._get_cache_path(ticker)
        try:
            data['cached_at'] = datetime.now().isoformat()
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


# =============================================================================
# Yahoo Finance Data Fetcher
# =============================================================================

class YahooFinanceFetcher:
    """Fetches fundamental data from Yahoo Finance"""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            self.available = False
    
    def get_stock(self, ticker: str):
        """Get yfinance Ticker object"""
        if not self.available:
            return None
        return self.yf.Ticker(ticker)
    
    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Fetch company metadata"""
        stock = self.get_stock(ticker)
        if not stock:
            return None
        
        try:
            info = stock.info
            return CompanyInfo(
                ticker=ticker.upper(),
                name=info.get('longName', info.get('shortName', ticker)),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                employees=info.get('fullTimeEmployees', 0),
                description=info.get('longBusinessSummary', ''),
                website=info.get('website', ''),
                country=info.get('country', 'Unknown'),
                exchange=info.get('exchange', 'Unknown'),
                currency=info.get('currency', 'USD')
            )
        except Exception as e:
            logger.error(f"Error fetching company info: {e}")
            return None
    
    def get_income_statement(self, ticker: str, quarterly: bool = False) -> List[Dict]:
        """Fetch income statement data"""
        stock = self.get_stock(ticker)
        if not stock:
            return []
        
        try:
            if quarterly:
                df = stock.quarterly_income_stmt
            else:
                df = stock.income_stmt
            
            if df is None or df.empty:
                return []
            
            statements = []
            for col in df.columns:
                data = df[col].to_dict()
                # Clean NaN values
                data = {k: (float(v) if v == v else 0.0) for k, v in data.items()}
                statements.append({
                    'period': 'quarterly' if quarterly else 'annual',
                    'date': col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col),
                    'data': data
                })
            return statements
        except Exception as e:
            logger.error(f"Error fetching income statement: {e}")
            return []
    
    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> List[Dict]:
        """Fetch balance sheet data"""
        stock = self.get_stock(ticker)
        if not stock:
            return []
        
        try:
            if quarterly:
                df = stock.quarterly_balance_sheet
            else:
                df = stock.balance_sheet
            
            if df is None or df.empty:
                return []
            
            statements = []
            for col in df.columns:
                data = df[col].to_dict()
                data = {k: (float(v) if v == v else 0.0) for k, v in data.items()}
                statements.append({
                    'period': 'quarterly' if quarterly else 'annual',
                    'date': col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col),
                    'data': data
                })
            return statements
        except Exception as e:
            logger.error(f"Error fetching balance sheet: {e}")
            return []
    
    def get_cash_flow(self, ticker: str, quarterly: bool = False) -> List[Dict]:
        """Fetch cash flow statement data"""
        stock = self.get_stock(ticker)
        if not stock:
            return []
        
        try:
            if quarterly:
                df = stock.quarterly_cashflow
            else:
                df = stock.cashflow
            
            if df is None or df.empty:
                return []
            
            statements = []
            for col in df.columns:
                data = df[col].to_dict()
                data = {k: (float(v) if v == v else 0.0) for k, v in data.items()}
                statements.append({
                    'period': 'quarterly' if quarterly else 'annual',
                    'date': col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col),
                    'data': data
                })
            return statements
        except Exception as e:
            logger.error(f"Error fetching cash flow: {e}")
            return []
    
    def get_key_stats(self, ticker: str) -> Dict:
        """Fetch key statistics from Yahoo Finance"""
        stock = self.get_stock(ticker)
        if not stock:
            return {}
        
        try:
            info = stock.info
            return {
                # Valuation
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_value': info.get('enterpriseValue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                
                # Profitability
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                
                # Dividend
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield'),
                
                # Financial Health
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                
                # Share Statistics
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'shares_short': info.get('sharesShort'),
                'short_ratio': info.get('shortRatio'),
                'short_percent_of_float': info.get('shortPercentOfFloat'),
                
                # Price Info
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'target_high_price': info.get('targetHighPrice'),
                'target_low_price': info.get('targetLowPrice'),
                'target_mean_price': info.get('targetMeanPrice'),
                'recommendation_mean': info.get('recommendationMean'),
                'recommendation_key': info.get('recommendationKey'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions'),
                
                # Other
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_day_average': info.get('fiftyDayAverage'),
                'two_hundred_day_average': info.get('twoHundredDayAverage'),
            }
        except Exception as e:
            logger.error(f"Error fetching key stats: {e}")
            return {}
    
    def get_historical_data(self, ticker: str, period: str = "5y") -> Dict:
        """Fetch historical price data for trend analysis"""
        stock = self.get_stock(ticker)
        if not stock:
            return {}
        
        try:
            hist = stock.history(period=period)
            if hist.empty:
                return {}
            
            return {
                'prices': hist['Close'].tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in hist.index],
                'volumes': hist['Volume'].tolist(),
                'high_52w': hist['Close'].tail(252).max() if len(hist) >= 252 else hist['Close'].max(),
                'low_52w': hist['Close'].tail(252).min() if len(hist) >= 252 else hist['Close'].min(),
                'avg_volume_30d': hist['Volume'].tail(30).mean() if len(hist) >= 30 else hist['Volume'].mean(),
            }
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {}
    
    def get_peer_tickers(self, ticker: str) -> List[str]:
        """Get peer company tickers for comparison"""
        stock = self.get_stock(ticker)
        if not stock:
            return []
        
        try:
            info = stock.info
            # Try to get recommendations which often include peers
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                # Extract unique tickers from recommendations (if available)
                pass
            
            # Fallback: Use sector-based ETF components or predefined mappings
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            
            # Common sector ETF tickers for peer comparison
            sector_etfs = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'CRM', 'ADBE', 'INTC', 'AMD', 'ORCL'],
                'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
                'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG'],
                'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'KMB', 'GIS'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
                'Industrials': ['HON', 'UPS', 'UNP', 'BA', 'CAT', 'GE', 'MMM', 'LMT', 'RTX', 'DE'],
                'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'WELL', 'AVB', 'EQR', 'DLR'],
                'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'PPG'],
            }
            
            peers = sector_etfs.get(sector, [])
            # Remove the ticker itself from peers
            peers = [p for p in peers if p.upper() != ticker.upper()]
            return peers[:5]  # Return top 5 peers
            
        except Exception as e:
            logger.error(f"Error getting peer tickers: {e}")
            return []


# =============================================================================
# SEC EDGAR Data Fetcher
# =============================================================================

class SECEdgarFetcher:
    """Fetches data from SEC EDGAR (optional, enhances data quality)"""
    
    BASE_URL = "https://data.sec.gov"
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Investment Analysis Tool contact@example.com',
            'Accept-Encoding': 'gzip, deflate'
        }
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number from ticker"""
        try:
            url = f"{self.BASE_URL}/submissions/CIK{ticker.upper()}.json"
            # Try ticker lookup
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(tickers_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for entry in data.values():
                    if entry.get('ticker', '').upper() == ticker.upper():
                        return str(entry.get('cik_str', '')).zfill(10)
            return None
        except Exception as e:
            logger.error(f"Error getting CIK: {e}")
            return None
    
    def get_company_facts(self, ticker: str) -> Dict:
        """Get company facts from SEC EDGAR"""
        cik = self.get_cik(ticker)
        if not cik:
            return {}
        
        try:
            url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Error fetching SEC data: {e}")
            return {}
    
    def get_recent_filings(self, ticker: str, filing_type: str = "10-K", count: int = 5) -> List[Dict]:
        """Get recent SEC filings metadata"""
        cik = self.get_cik(ticker)
        if not cik:
            return []
        
        try:
            url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code != 200:
                return []
            
            data = response.json()
            filings = []
            recent = data.get('filings', {}).get('recent', {})
            
            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            
            for i, form in enumerate(forms):
                if form == filing_type and len(filings) < count:
                    filings.append({
                        'form': form,
                        'date': dates[i] if i < len(dates) else '',
                        'accession': accessions[i] if i < len(accessions) else ''
                    })
            
            return filings
        except Exception as e:
            logger.error(f"Error fetching SEC filings: {e}")
            return []


# =============================================================================
# Ratio Calculator
# =============================================================================

class RatioCalculator:
    """Calculates financial ratios from raw financial statement data"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division handling zero and None values"""
        try:
            if denominator is None or denominator == 0:
                return default
            if numerator is None:
                return default
            return numerator / denominator
        except:
            return default
    
    @staticmethod
    def get_value(data: Dict, keys: List[str], default: float = 0.0) -> float:
        """Get value from dict trying multiple possible keys"""
        for key in keys:
            if key in data and data[key] is not None and data[key] == data[key]:  # Check for NaN
                return float(data[key])
        return default
    
    def calculate_valuation_ratios(self, info: Dict, income: Dict, balance: Dict, 
                                   market_cap: float, price: float) -> Dict[str, float]:
        """Calculate valuation ratios"""
        
        # Get values with fallbacks for different naming conventions
        net_income = self.get_value(income, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
        revenue = self.get_value(income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
        ebitda = self.get_value(income, ['EBITDA', 'Ebitda', 'Normalized EBITDA'])
        
        total_equity = self.get_value(balance, ['Total Stockholder Equity', 'Stockholders Equity', 
                                                 'Total Equity Gross Minority Interest', 'Common Stock Equity'])
        total_assets = self.get_value(balance, ['Total Assets', 'TotalAssets'])
        total_debt = self.get_value(balance, ['Total Debt', 'TotalDebt', 'Long Term Debt'])
        cash = self.get_value(balance, ['Cash And Cash Equivalents', 'Cash', 'Cash And Short Term Investments'])
        
        shares_outstanding = info.get('shares_outstanding', 0) or 1
        
        # Calculate EPS
        eps = self.safe_divide(net_income, shares_outstanding)
        
        # Calculate book value per share
        book_value_per_share = self.safe_divide(total_equity, shares_outstanding)
        
        # Enterprise Value
        enterprise_value = market_cap + total_debt - cash
        
        return {
            'price': price,
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'pe_ratio': self.safe_divide(price, eps) if eps > 0 else info.get('trailing_pe', 0),
            'forward_pe': info.get('forward_pe', 0),
            'peg_ratio': info.get('peg_ratio', 0),
            'price_to_book': self.safe_divide(price, book_value_per_share) if book_value_per_share > 0 else info.get('price_to_book', 0),
            'price_to_sales': self.safe_divide(market_cap, revenue) if revenue > 0 else info.get('price_to_sales', 0),
            'price_to_cash_flow': self.safe_divide(market_cap, info.get('operating_cash_flow', 0)),
            'price_to_free_cash_flow': self.safe_divide(market_cap, info.get('free_cash_flow', 0)),
            'ev_to_ebitda': self.safe_divide(enterprise_value, ebitda) if ebitda > 0 else info.get('ev_to_ebitda', 0),
            'ev_to_revenue': self.safe_divide(enterprise_value, revenue) if revenue > 0 else info.get('ev_to_revenue', 0),
            'ev_to_fcf': self.safe_divide(enterprise_value, info.get('free_cash_flow', 0)),
            'earnings_yield': self.safe_divide(eps, price) * 100,  # Inverse of P/E as percentage
            'fcf_yield': self.safe_divide(info.get('free_cash_flow', 0), market_cap) * 100,
            'book_value_per_share': book_value_per_share,
            'eps': eps,
        }
    
    def calculate_profitability_ratios(self, income: Dict, balance: Dict, 
                                       prev_balance: Dict = None) -> Dict[str, float]:
        """Calculate profitability ratios"""
        
        revenue = self.get_value(income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
        gross_profit = self.get_value(income, ['Gross Profit', 'GrossProfit'])
        operating_income = self.get_value(income, ['Operating Income', 'OperatingIncome', 'EBIT'])
        net_income = self.get_value(income, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
        ebitda = self.get_value(income, ['EBITDA', 'Ebitda', 'Normalized EBITDA'])
        
        total_equity = self.get_value(balance, ['Total Stockholder Equity', 'Stockholders Equity',
                                                 'Total Equity Gross Minority Interest'])
        total_assets = self.get_value(balance, ['Total Assets', 'TotalAssets'])
        
        # For ROIC, we need invested capital
        total_debt = self.get_value(balance, ['Total Debt', 'TotalDebt', 'Long Term Debt'])
        cash = self.get_value(balance, ['Cash And Cash Equivalents', 'Cash'])
        invested_capital = total_equity + total_debt - cash
        
        # Calculate NOPAT (Net Operating Profit After Tax)
        # Assume 25% tax rate if not available
        tax_rate = 0.25
        nopat = operating_income * (1 - tax_rate)
        
        # Average assets/equity if previous period available
        if prev_balance:
            prev_equity = self.get_value(prev_balance, ['Total Stockholder Equity', 'Stockholders Equity',
                                                         'Total Equity Gross Minority Interest'])
            prev_assets = self.get_value(prev_balance, ['Total Assets', 'TotalAssets'])
            avg_equity = (total_equity + prev_equity) / 2
            avg_assets = (total_assets + prev_assets) / 2
        else:
            avg_equity = total_equity
            avg_assets = total_assets
        
        return {
            'gross_margin': self.safe_divide(gross_profit, revenue) * 100,
            'operating_margin': self.safe_divide(operating_income, revenue) * 100,
            'net_profit_margin': self.safe_divide(net_income, revenue) * 100,
            'ebitda_margin': self.safe_divide(ebitda, revenue) * 100,
            'return_on_equity': self.safe_divide(net_income, avg_equity) * 100,
            'return_on_assets': self.safe_divide(net_income, avg_assets) * 100,
            'return_on_invested_capital': self.safe_divide(nopat, invested_capital) * 100,
            'return_on_capital_employed': self.safe_divide(operating_income, (total_assets - self.get_value(balance, ['Total Current Liabilities', 'Current Liabilities']))) * 100,
            'gross_profit': gross_profit,
            'operating_income': operating_income,
            'net_income': net_income,
            'ebitda': ebitda,
            'revenue': revenue,
        }
    
    def calculate_liquidity_ratios(self, balance: Dict) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        
        current_assets = self.get_value(balance, ['Total Current Assets', 'CurrentAssets', 'Current Assets'])
        current_liabilities = self.get_value(balance, ['Total Current Liabilities', 'CurrentLiabilities', 'Current Liabilities'])
        inventory = self.get_value(balance, ['Inventory', 'Inventories'])
        cash = self.get_value(balance, ['Cash And Cash Equivalents', 'Cash', 'Cash And Short Term Investments'])
        receivables = self.get_value(balance, ['Net Receivables', 'Accounts Receivable', 'Receivables'])
        
        return {
            'current_ratio': self.safe_divide(current_assets, current_liabilities),
            'quick_ratio': self.safe_divide(current_assets - inventory, current_liabilities),
            'cash_ratio': self.safe_divide(cash, current_liabilities),
            'working_capital': current_assets - current_liabilities,
            'working_capital_ratio': self.safe_divide(current_assets - current_liabilities, current_assets) * 100,
            'defensive_interval': self.safe_divide(current_assets, self.get_value(balance, ['Operating Expenses', 'Total Operating Expenses'], 1) / 365),
            'cash': cash,
            'current_assets': current_assets,
            'current_liabilities': current_liabilities,
            'inventory': inventory,
            'receivables': receivables,
        }
    
    def calculate_leverage_ratios(self, income: Dict, balance: Dict) -> Dict[str, float]:
        """Calculate leverage/solvency ratios"""
        
        total_debt = self.get_value(balance, ['Total Debt', 'TotalDebt'])
        long_term_debt = self.get_value(balance, ['Long Term Debt', 'LongTermDebt'])
        short_term_debt = self.get_value(balance, ['Short Term Debt', 'Short Long Term Debt', 'Current Debt'])
        total_equity = self.get_value(balance, ['Total Stockholder Equity', 'Stockholders Equity',
                                                 'Total Equity Gross Minority Interest'])
        total_assets = self.get_value(balance, ['Total Assets', 'TotalAssets'])
        total_liabilities = self.get_value(balance, ['Total Liabilities', 'Total Liabilities Net Minority Interest'])
        
        operating_income = self.get_value(income, ['Operating Income', 'OperatingIncome', 'EBIT'])
        interest_expense = self.get_value(income, ['Interest Expense', 'InterestExpense'])
        ebitda = self.get_value(income, ['EBITDA', 'Ebitda'])
        
        return {
            'debt_to_equity': self.safe_divide(total_debt, total_equity),
            'debt_to_assets': self.safe_divide(total_debt, total_assets),
            'debt_to_capital': self.safe_divide(total_debt, total_debt + total_equity),
            'long_term_debt_to_equity': self.safe_divide(long_term_debt, total_equity),
            'equity_multiplier': self.safe_divide(total_assets, total_equity),
            'financial_leverage': self.safe_divide(total_assets, total_equity),
            'interest_coverage': self.safe_divide(operating_income, interest_expense) if interest_expense > 0 else 999,
            'debt_service_coverage': self.safe_divide(ebitda, interest_expense + short_term_debt) if (interest_expense + short_term_debt) > 0 else 999,
            'total_debt': total_debt,
            'long_term_debt': long_term_debt,
            'short_term_debt': short_term_debt,
            'total_equity': total_equity,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
        }
    
    def calculate_efficiency_ratios(self, income: Dict, balance: Dict, 
                                    prev_balance: Dict = None) -> Dict[str, float]:
        """Calculate efficiency/activity ratios"""
        
        revenue = self.get_value(income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
        cogs = self.get_value(income, ['Cost Of Revenue', 'CostOfRevenue', 'Cost of Goods Sold'])
        
        total_assets = self.get_value(balance, ['Total Assets', 'TotalAssets'])
        inventory = self.get_value(balance, ['Inventory', 'Inventories'])
        receivables = self.get_value(balance, ['Net Receivables', 'Accounts Receivable'])
        payables = self.get_value(balance, ['Accounts Payable', 'AccountsPayable'])
        fixed_assets = self.get_value(balance, ['Net PPE', 'Property Plant Equipment Net', 'Fixed Assets'])
        
        # Average values if previous period available
        if prev_balance:
            avg_inventory = (inventory + self.get_value(prev_balance, ['Inventory', 'Inventories'])) / 2
            avg_receivables = (receivables + self.get_value(prev_balance, ['Net Receivables', 'Accounts Receivable'])) / 2
            avg_payables = (payables + self.get_value(prev_balance, ['Accounts Payable', 'AccountsPayable'])) / 2
            avg_assets = (total_assets + self.get_value(prev_balance, ['Total Assets', 'TotalAssets'])) / 2
        else:
            avg_inventory = inventory
            avg_receivables = receivables
            avg_payables = payables
            avg_assets = total_assets
        
        # Turnover ratios
        inventory_turnover = self.safe_divide(cogs, avg_inventory)
        receivables_turnover = self.safe_divide(revenue, avg_receivables)
        payables_turnover = self.safe_divide(cogs, avg_payables)
        asset_turnover = self.safe_divide(revenue, avg_assets)
        
        # Days ratios
        days_inventory = self.safe_divide(365, inventory_turnover) if inventory_turnover > 0 else 0
        days_receivables = self.safe_divide(365, receivables_turnover) if receivables_turnover > 0 else 0
        days_payables = self.safe_divide(365, payables_turnover) if payables_turnover > 0 else 0
        
        # Cash conversion cycle
        cash_conversion_cycle = days_inventory + days_receivables - days_payables
        
        return {
            'asset_turnover': asset_turnover,
            'inventory_turnover': inventory_turnover,
            'receivables_turnover': receivables_turnover,
            'payables_turnover': payables_turnover,
            'fixed_asset_turnover': self.safe_divide(revenue, fixed_assets),
            'days_inventory_outstanding': days_inventory,
            'days_sales_outstanding': days_receivables,
            'days_payables_outstanding': days_payables,
            'cash_conversion_cycle': cash_conversion_cycle,
            'operating_cycle': days_inventory + days_receivables,
        }
    
    def calculate_growth_metrics(self, income_statements: List[Dict], 
                                 balance_sheets: List[Dict]) -> Dict[str, float]:
        """Calculate growth metrics from historical data"""
        
        if len(income_statements) < 2:
            return {
                'revenue_growth_yoy': 0,
                'revenue_growth_3y_cagr': 0,
                'earnings_growth_yoy': 0,
                'earnings_growth_3y_cagr': 0,
                'ebitda_growth_yoy': 0,
                'book_value_growth_yoy': 0,
                'eps_growth_yoy': 0,
            }
        
        def cagr(start_value: float, end_value: float, years: int) -> float:
            """Calculate Compound Annual Growth Rate"""
            if start_value <= 0 or end_value <= 0 or years <= 0:
                return 0
            return (pow(end_value / start_value, 1 / years) - 1) * 100
        
        def yoy_growth(current: float, previous: float) -> float:
            """Calculate Year-over-Year growth"""
            if previous == 0:
                return 0
            return ((current - previous) / abs(previous)) * 100
        
        # Get current and previous period data
        current_income = income_statements[0].get('data', {})
        prev_income = income_statements[1].get('data', {})
        
        current_revenue = self.get_value(current_income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
        prev_revenue = self.get_value(prev_income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
        
        current_net_income = self.get_value(current_income, ['Net Income', 'NetIncome'])
        prev_net_income = self.get_value(prev_income, ['Net Income', 'NetIncome'])
        
        current_ebitda = self.get_value(current_income, ['EBITDA', 'Ebitda'])
        prev_ebitda = self.get_value(prev_income, ['EBITDA', 'Ebitda'])
        
        # Balance sheet growth
        current_equity = self.get_value(balance_sheets[0].get('data', {}), 
                                        ['Total Stockholder Equity', 'Stockholders Equity'])
        prev_equity = self.get_value(balance_sheets[1].get('data', {}), 
                                     ['Total Stockholder Equity', 'Stockholders Equity']) if len(balance_sheets) > 1 else 0
        
        # Calculate 3-year CAGR if data available
        revenue_cagr_3y = 0
        earnings_cagr_3y = 0
        if len(income_statements) >= 4:
            oldest_income = income_statements[3].get('data', {})
            oldest_revenue = self.get_value(oldest_income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
            oldest_net_income = self.get_value(oldest_income, ['Net Income', 'NetIncome'])
            revenue_cagr_3y = cagr(oldest_revenue, current_revenue, 3)
            earnings_cagr_3y = cagr(oldest_net_income, current_net_income, 3) if oldest_net_income > 0 and current_net_income > 0 else 0
        
        return {
            'revenue_growth_yoy': yoy_growth(current_revenue, prev_revenue),
            'revenue_growth_3y_cagr': revenue_cagr_3y,
            'earnings_growth_yoy': yoy_growth(current_net_income, prev_net_income),
            'earnings_growth_3y_cagr': earnings_cagr_3y,
            'ebitda_growth_yoy': yoy_growth(current_ebitda, prev_ebitda),
            'book_value_growth_yoy': yoy_growth(current_equity, prev_equity),
            'gross_profit_growth_yoy': yoy_growth(
                self.get_value(current_income, ['Gross Profit', 'GrossProfit']),
                self.get_value(prev_income, ['Gross Profit', 'GrossProfit'])
            ),
            'operating_income_growth_yoy': yoy_growth(
                self.get_value(current_income, ['Operating Income', 'OperatingIncome']),
                self.get_value(prev_income, ['Operating Income', 'OperatingIncome'])
            ),
        }
    
    def calculate_dividend_metrics(self, info: Dict, income: Dict, cash_flow: Dict) -> Dict[str, Any]:
        """Calculate dividend-related metrics"""
        
        dividend_yield = info.get('dividend_yield', 0) or 0
        dividend_rate = info.get('dividend_rate', 0) or 0
        payout_ratio = info.get('payout_ratio', 0) or 0
        
        net_income = self.get_value(income, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
        free_cash_flow = self.get_value(cash_flow, ['Free Cash Flow', 'FreeCashFlow'])
        dividends_paid = abs(self.get_value(cash_flow, ['Cash Dividends Paid', 'Dividends Paid', 'Common Stock Dividend Paid']))
        
        # Calculate additional dividend metrics
        dividend_coverage = self.safe_divide(net_income, dividends_paid) if dividends_paid > 0 else 0
        fcf_dividend_coverage = self.safe_divide(free_cash_flow, dividends_paid) if dividends_paid > 0 else 0
        
        return {
            'dividend_yield': dividend_yield * 100 if dividend_yield < 1 else dividend_yield,  # Convert to percentage
            'dividend_rate': dividend_rate,
            'payout_ratio': payout_ratio * 100 if payout_ratio < 1 else payout_ratio,
            'dividend_coverage': dividend_coverage,
            'fcf_dividend_coverage': fcf_dividend_coverage,
            'dividends_paid': dividends_paid,
            'five_year_avg_yield': info.get('five_year_avg_dividend_yield', 0) or 0,
            'is_dividend_payer': dividends_paid > 0 or dividend_rate > 0,
            'dividend_sustainability_score': min(100, (dividend_coverage * 20 + fcf_dividend_coverage * 20 + 
                                                        (1 if payout_ratio < 0.6 else 0) * 30 + 
                                                        (1 if dividend_yield > 0 else 0) * 30)),
        }
    
    def calculate_quality_scores(self, income: Dict, balance: Dict, cash_flow: Dict,
                                 income_statements: List[Dict] = None) -> Dict[str, Any]:
        """Calculate quality scores (Altman Z-Score, Piotroski F-Score, etc.)"""
        
        # === Altman Z-Score ===
        # Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        # A = Working Capital / Total Assets
        # B = Retained Earnings / Total Assets
        # C = EBIT / Total Assets
        # D = Market Value of Equity / Total Liabilities
        # E = Sales / Total Assets
        
        total_assets = self.get_value(balance, ['Total Assets', 'TotalAssets'])
        current_assets = self.get_value(balance, ['Total Current Assets', 'CurrentAssets'])
        current_liabilities = self.get_value(balance, ['Total Current Liabilities', 'CurrentLiabilities'])
        retained_earnings = self.get_value(balance, ['Retained Earnings', 'RetainedEarnings'])
        total_liabilities = self.get_value(balance, ['Total Liabilities', 'Total Liabilities Net Minority Interest'])
        total_equity = self.get_value(balance, ['Total Stockholder Equity', 'Stockholders Equity'])
        
        operating_income = self.get_value(income, ['Operating Income', 'OperatingIncome', 'EBIT'])
        revenue = self.get_value(income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
        net_income = self.get_value(income, ['Net Income', 'NetIncome'])
        
        working_capital = current_assets - current_liabilities
        
        if total_assets > 0 and total_liabilities > 0:
            z_a = self.safe_divide(working_capital, total_assets)
            z_b = self.safe_divide(retained_earnings, total_assets)
            z_c = self.safe_divide(operating_income, total_assets)
            z_d = self.safe_divide(total_equity, total_liabilities)  # Using book value as proxy
            z_e = self.safe_divide(revenue, total_assets)
            
            altman_z = 1.2 * z_a + 1.4 * z_b + 3.3 * z_c + 0.6 * z_d + 1.0 * z_e
        else:
            altman_z = 0
        
        # Z-Score interpretation
        if altman_z > 2.99:
            z_interpretation = "Safe Zone"
        elif altman_z > 1.81:
            z_interpretation = "Grey Zone"
        else:
            z_interpretation = "Distress Zone"
        
        # === Piotroski F-Score (0-9) ===
        # Profitability (4 points)
        # 1. ROA > 0
        # 2. Operating Cash Flow > 0
        # 3. ROA increasing
        # 4. CFO > Net Income (quality of earnings)
        
        # Leverage/Liquidity (3 points)
        # 5. Decrease in Long-term Debt
        # 6. Current Ratio increasing
        # 7. No new shares issued
        
        # Operating Efficiency (2 points)
        # 8. Gross Margin increasing
        # 9. Asset Turnover increasing
        
        operating_cash_flow = self.get_value(cash_flow, ['Operating Cash Flow', 'Cash Flow From Operating Activities'])
        
        f_score = 0
        
        # Profitability signals
        roa = self.safe_divide(net_income, total_assets)
        if roa > 0:
            f_score += 1
        if operating_cash_flow > 0:
            f_score += 1
        if operating_cash_flow > net_income:  # Quality of earnings
            f_score += 1
        
        # Leverage signals
        current_ratio = self.safe_divide(current_assets, current_liabilities)
        if current_ratio > 1:
            f_score += 1
        
        # For full F-Score, would need historical data comparison
        # Adding partial scores based on current data
        gross_margin = self.safe_divide(self.get_value(income, ['Gross Profit', 'GrossProfit']), revenue)
        if gross_margin > 0.3:  # Above 30% gross margin
            f_score += 1
        
        asset_turnover = self.safe_divide(revenue, total_assets)
        if asset_turnover > 0.5:
            f_score += 1
        
        # Historical comparison for additional points
        if income_statements and len(income_statements) >= 2:
            prev_income = income_statements[1].get('data', {})
            prev_roa = self.safe_divide(
                self.get_value(prev_income, ['Net Income', 'NetIncome']),
                total_assets
            )
            if roa > prev_roa:
                f_score += 1
            
            prev_gross_margin = self.safe_divide(
                self.get_value(prev_income, ['Gross Profit', 'GrossProfit']),
                self.get_value(prev_income, ['Total Revenue', 'TotalRevenue', 'Revenue'])
            )
            if gross_margin > prev_gross_margin:
                f_score += 1
        
        # Cap at 9
        f_score = min(9, f_score)
        
        # F-Score interpretation
        if f_score >= 7:
            f_interpretation = "Strong"
        elif f_score >= 4:
            f_interpretation = "Neutral"
        else:
            f_interpretation = "Weak"
        
        # === DuPont Analysis ===
        # ROE = Net Profit Margin × Asset Turnover × Equity Multiplier
        net_margin = self.safe_divide(net_income, revenue)
        equity_multiplier = self.safe_divide(total_assets, total_equity)
        dupont_roe = net_margin * asset_turnover * equity_multiplier * 100
        
        # === Beneish M-Score (Earnings Manipulation Detection) ===
        # Simplified version - would need more historical data for full calculation
        # M > -1.78 suggests possible manipulation
        
        return {
            'altman_z_score': round(altman_z, 2),
            'altman_z_interpretation': z_interpretation,
            'piotroski_f_score': f_score,
            'piotroski_f_interpretation': f_interpretation,
            'dupont_roe': round(dupont_roe, 2),
            'dupont_net_margin': round(net_margin * 100, 2),
            'dupont_asset_turnover': round(asset_turnover, 2),
            'dupont_equity_multiplier': round(equity_multiplier, 2),
            'accruals_ratio': round(self.safe_divide(net_income - operating_cash_flow, total_assets) * 100, 2),
            'quality_of_earnings': round(self.safe_divide(operating_cash_flow, net_income), 2) if net_income != 0 else 0,
        }


# =============================================================================
# Main Fundamental Calculator
# =============================================================================

class FundamentalCalculator:
    """
    Main class for fetching and calculating fundamental data.
    Aggregates data from multiple sources and calculates comprehensive metrics.
    """
    
    def __init__(self, use_cache: bool = True):
        self.yahoo = YahooFinanceFetcher()
        self.sec = SECEdgarFetcher()
        self.ratio_calc = RatioCalculator()
        self.cache = FundamentalCache() if use_cache else None
    
    def get_company_info(self, ticker: str) -> Dict:
        """Get company metadata"""
        info = self.yahoo.get_company_info(ticker)
        if info:
            return {
                'ticker': info.ticker,
                'name': info.name,
                'sector': info.sector,
                'industry': info.industry,
                'market_cap': info.market_cap,
                'employees': info.employees,
                'description': info.description[:500] + '...' if len(info.description) > 500 else info.description,
                'website': info.website,
                'country': info.country,
                'exchange': info.exchange,
                'currency': info.currency,
            }
        return {}
    
    def get_financial_statements(self, ticker: str) -> Dict:
        """Get all financial statements"""
        return {
            'income_statement_annual': self.yahoo.get_income_statement(ticker, quarterly=False),
            'income_statement_quarterly': self.yahoo.get_income_statement(ticker, quarterly=True),
            'balance_sheet_annual': self.yahoo.get_balance_sheet(ticker, quarterly=False),
            'balance_sheet_quarterly': self.yahoo.get_balance_sheet(ticker, quarterly=True),
            'cash_flow_annual': self.yahoo.get_cash_flow(ticker, quarterly=False),
            'cash_flow_quarterly': self.yahoo.get_cash_flow(ticker, quarterly=True),
        }
    
    def get_peer_comparison(self, ticker: str, peers: List[str] = None) -> Dict:
        """Get peer comparison data"""
        if not peers:
            peers = self.yahoo.get_peer_tickers(ticker)
        
        if not peers:
            return {'peers': [], 'comparison': {}}
        
        peer_data = {}
        metrics_to_compare = ['trailing_pe', 'price_to_book', 'profit_margin', 
                             'return_on_equity', 'debt_to_equity', 'revenue_growth']
        
        # Get target company data
        target_stats = self.yahoo.get_key_stats(ticker)
        peer_data[ticker] = {m: target_stats.get(m) for m in metrics_to_compare}
        
        # Get peer data
        for peer in peers[:5]:  # Limit to 5 peers
            try:
                stats = self.yahoo.get_key_stats(peer)
                peer_data[peer] = {m: stats.get(m) for m in metrics_to_compare}
            except:
                continue
        
        # Calculate averages (excluding target)
        averages = {}
        for metric in metrics_to_compare:
            values = [peer_data[p].get(metric) for p in peers if p in peer_data and peer_data[p].get(metric)]
            values = [v for v in values if v is not None and v == v]  # Remove None and NaN
            if values:
                averages[metric] = sum(values) / len(values)
        
        return {
            'peers': peers,
            'peer_metrics': peer_data,
            'peer_averages': averages,
            'vs_peers': {
                metric: {
                    'value': target_stats.get(metric),
                    'peer_avg': averages.get(metric),
                    'percentile': self._calculate_percentile(target_stats.get(metric), 
                                                              [peer_data[p].get(metric) for p in peers if p in peer_data])
                }
                for metric in metrics_to_compare
            }
        }
    
    def _calculate_percentile(self, value: float, comparison_values: List[float]) -> float:
        """Calculate what percentile a value falls in"""
        if value is None or not comparison_values:
            return 50.0
        comparison_values = [v for v in comparison_values if v is not None and v == v]
        if not comparison_values:
            return 50.0
        below = sum(1 for v in comparison_values if v < value)
        return (below / len(comparison_values)) * 100
    
    def get_historical_valuation(self, ticker: str) -> Dict:
        """Get historical valuation data for comparison"""
        stats = self.yahoo.get_key_stats(ticker)
        hist = self.yahoo.get_historical_data(ticker)
        
        # Current vs historical averages
        current_price = stats.get('current_price', 0)
        fifty_day_avg = stats.get('fifty_day_average', 0)
        two_hundred_day_avg = stats.get('two_hundred_day_average', 0)
        fifty_two_week_high = stats.get('fifty_two_week_high', 0)
        fifty_two_week_low = stats.get('fifty_two_week_low', 0)
        
        return {
            'current_price': current_price,
            'fifty_day_average': fifty_day_avg,
            'two_hundred_day_average': two_hundred_day_avg,
            'fifty_two_week_high': fifty_two_week_high,
            'fifty_two_week_low': fifty_two_week_low,
            'price_vs_50d_avg': ((current_price - fifty_day_avg) / fifty_day_avg * 100) if fifty_day_avg else 0,
            'price_vs_200d_avg': ((current_price - two_hundred_day_avg) / two_hundred_day_avg * 100) if two_hundred_day_avg else 0,
            'distance_from_52w_high': ((fifty_two_week_high - current_price) / fifty_two_week_high * 100) if fifty_two_week_high else 0,
            'distance_from_52w_low': ((current_price - fifty_two_week_low) / fifty_two_week_low * 100) if fifty_two_week_low else 0,
            '52w_range_position': ((current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low) * 100) if (fifty_two_week_high - fifty_two_week_low) > 0 else 50,
        }
    
    def get_sec_filings(self, ticker: str) -> Dict:
        """Get SEC filing information"""
        filings_10k = self.sec.get_recent_filings(ticker, '10-K', 3)
        filings_10q = self.sec.get_recent_filings(ticker, '10-Q', 4)
        filings_8k = self.sec.get_recent_filings(ticker, '8-K', 5)
        
        return {
            'recent_10k': filings_10k,
            'recent_10q': filings_10q,
            'recent_8k': filings_8k,
            'has_recent_filings': len(filings_10k) > 0 or len(filings_10q) > 0,
        }
    
    def get_all_fundamentals(self, ticker: str, include_peers: bool = True, 
                            include_sec: bool = False) -> Dict:
        """
        Get comprehensive fundamental data for a ticker.
        This is the main method to call for complete analysis.
        """
        ticker = ticker.upper()
        
        # Check cache
        if self.cache:
            cached = self.cache.get(ticker)
            if cached:
                return cached
        
        logger.info(f"Fetching fundamental data for {ticker}")
        
        # Fetch raw data
        company_info = self.get_company_info(ticker)
        statements = self.get_financial_statements(ticker)
        key_stats = self.yahoo.get_key_stats(ticker)
        historical = self.get_historical_valuation(ticker)
        
        # Get most recent statements for calculations
        income_annual = statements['income_statement_annual']
        balance_annual = statements['balance_sheet_annual']
        cash_flow_annual = statements['cash_flow_annual']
        
        current_income = income_annual[0].get('data', {}) if income_annual else {}
        prev_income = income_annual[1].get('data', {}) if len(income_annual) > 1 else {}
        current_balance = balance_annual[0].get('data', {}) if balance_annual else {}
        prev_balance = balance_annual[1].get('data', {}) if len(balance_annual) > 1 else {}
        current_cash_flow = cash_flow_annual[0].get('data', {}) if cash_flow_annual else {}
        
        # Get market data
        market_cap = company_info.get('market_cap', 0) or key_stats.get('market_cap', 0) or 0
        current_price = key_stats.get('current_price', 0) or 0
        
        # Calculate all ratios
        valuation = self.ratio_calc.calculate_valuation_ratios(
            key_stats, current_income, current_balance, market_cap, current_price
        )
        
        profitability = self.ratio_calc.calculate_profitability_ratios(
            current_income, current_balance, prev_balance
        )
        
        liquidity = self.ratio_calc.calculate_liquidity_ratios(current_balance)
        
        leverage = self.ratio_calc.calculate_leverage_ratios(current_income, current_balance)
        
        efficiency = self.ratio_calc.calculate_efficiency_ratios(
            current_income, current_balance, prev_balance
        )
        
        growth = self.ratio_calc.calculate_growth_metrics(income_annual, balance_annual)
        
        dividends = self.ratio_calc.calculate_dividend_metrics(
            key_stats, current_income, current_cash_flow
        )
        
        quality = self.ratio_calc.calculate_quality_scores(
            current_income, current_balance, current_cash_flow, income_annual
        )
        
        # Peer comparison
        peer_comparison = {}
        if include_peers:
            peer_comparison = self.get_peer_comparison(ticker)
        
        # SEC filings
        sec_filings = {}
        if include_sec:
            sec_filings = self.get_sec_filings(ticker)
        
        # Compile result
        result = {
            'ticker': ticker,
            'fetch_time': datetime.now().isoformat(),
            'company_info': company_info,
            'key_stats': key_stats,
            'financial_statements': {
                'income_annual': income_annual[:4] if income_annual else [],
                'income_quarterly': statements['income_statement_quarterly'][:4] if statements['income_statement_quarterly'] else [],
                'balance_annual': balance_annual[:4] if balance_annual else [],
                'balance_quarterly': statements['balance_sheet_quarterly'][:4] if statements['balance_sheet_quarterly'] else [],
                'cash_flow_annual': cash_flow_annual[:4] if cash_flow_annual else [],
                'cash_flow_quarterly': statements['cash_flow_quarterly'][:4] if statements['cash_flow_quarterly'] else [],
            },
            'valuation_ratios': valuation,
            'profitability_ratios': profitability,
            'liquidity_ratios': liquidity,
            'leverage_ratios': leverage,
            'efficiency_ratios': efficiency,
            'growth_metrics': growth,
            'dividend_metrics': dividends,
            'quality_scores': quality,
            'historical_valuation': historical,
            'peer_comparison': peer_comparison,
            'sec_filings': sec_filings,
        }
        
        # Cache result
        if self.cache:
            self.cache.set(ticker, result)
        
        return result
    
    def get_formatted_for_llm(self, ticker: str, max_length: int = 8000) -> str:
        """
        Get fundamental data formatted for LLM analysis.
        Returns a structured text summary of key metrics.
        """
        data = self.get_all_fundamentals(ticker)
        
        sections = []
        
        # Company Overview
        info = data.get('company_info', {})
        sections.append(f"""
=== COMPANY OVERVIEW ===
Ticker: {ticker}
Name: {info.get('name', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Market Cap: ${info.get('market_cap', 0):,.0f}
Country: {info.get('country', 'N/A')}
Employees: {info.get('employees', 0):,}
""")
        
        # Valuation Metrics
        val = data.get('valuation_ratios', {})
        sections.append(f"""
=== VALUATION METRICS ===
Current Price: ${val.get('price', 0):.2f}
Market Cap: ${val.get('market_cap', 0):,.0f}
Enterprise Value: ${val.get('enterprise_value', 0):,.0f}
P/E Ratio (TTM): {val.get('pe_ratio', 0):.2f}
Forward P/E: {val.get('forward_pe', 0):.2f}
PEG Ratio: {val.get('peg_ratio', 0):.2f}
Price/Book: {val.get('price_to_book', 0):.2f}
Price/Sales: {val.get('price_to_sales', 0):.2f}
EV/EBITDA: {val.get('ev_to_ebitda', 0):.2f}
EV/Revenue: {val.get('ev_to_revenue', 0):.2f}
Earnings Yield: {val.get('earnings_yield', 0):.2f}%
FCF Yield: {val.get('fcf_yield', 0):.2f}%
EPS: ${val.get('eps', 0):.2f}
Book Value/Share: ${val.get('book_value_per_share', 0):.2f}
""")
        
        # Profitability Metrics
        prof = data.get('profitability_ratios', {})
        sections.append(f"""
=== PROFITABILITY METRICS ===
Gross Margin: {prof.get('gross_margin', 0):.2f}%
Operating Margin: {prof.get('operating_margin', 0):.2f}%
Net Profit Margin: {prof.get('net_profit_margin', 0):.2f}%
EBITDA Margin: {prof.get('ebitda_margin', 0):.2f}%
Return on Equity (ROE): {prof.get('return_on_equity', 0):.2f}%
Return on Assets (ROA): {prof.get('return_on_assets', 0):.2f}%
Return on Invested Capital (ROIC): {prof.get('return_on_invested_capital', 0):.2f}%
Revenue: ${prof.get('revenue', 0):,.0f}
Net Income: ${prof.get('net_income', 0):,.0f}
EBITDA: ${prof.get('ebitda', 0):,.0f}
""")
        
        # Liquidity Metrics
        liq = data.get('liquidity_ratios', {})
        sections.append(f"""
=== LIQUIDITY METRICS ===
Current Ratio: {liq.get('current_ratio', 0):.2f}
Quick Ratio: {liq.get('quick_ratio', 0):.2f}
Cash Ratio: {liq.get('cash_ratio', 0):.2f}
Working Capital: ${liq.get('working_capital', 0):,.0f}
Cash & Equivalents: ${liq.get('cash', 0):,.0f}
""")
        
        # Leverage Metrics
        lev = data.get('leverage_ratios', {})
        sections.append(f"""
=== LEVERAGE/SOLVENCY METRICS ===
Debt/Equity: {lev.get('debt_to_equity', 0):.2f}
Debt/Assets: {lev.get('debt_to_assets', 0):.2f}
Interest Coverage: {lev.get('interest_coverage', 0):.2f}x
Equity Multiplier: {lev.get('equity_multiplier', 0):.2f}
Total Debt: ${lev.get('total_debt', 0):,.0f}
Total Equity: ${lev.get('total_equity', 0):,.0f}
""")
        
        # Efficiency Metrics
        eff = data.get('efficiency_ratios', {})
        sections.append(f"""
=== EFFICIENCY METRICS ===
Asset Turnover: {eff.get('asset_turnover', 0):.2f}
Inventory Turnover: {eff.get('inventory_turnover', 0):.2f}
Receivables Turnover: {eff.get('receivables_turnover', 0):.2f}
Days Inventory: {eff.get('days_inventory_outstanding', 0):.1f}
Days Sales Outstanding: {eff.get('days_sales_outstanding', 0):.1f}
Cash Conversion Cycle: {eff.get('cash_conversion_cycle', 0):.1f} days
""")
        
        # Growth Metrics
        growth = data.get('growth_metrics', {})
        sections.append(f"""
=== GROWTH METRICS ===
Revenue Growth (YoY): {growth.get('revenue_growth_yoy', 0):.2f}%
Revenue Growth (3Y CAGR): {growth.get('revenue_growth_3y_cagr', 0):.2f}%
Earnings Growth (YoY): {growth.get('earnings_growth_yoy', 0):.2f}%
Earnings Growth (3Y CAGR): {growth.get('earnings_growth_3y_cagr', 0):.2f}%
EBITDA Growth (YoY): {growth.get('ebitda_growth_yoy', 0):.2f}%
Book Value Growth (YoY): {growth.get('book_value_growth_yoy', 0):.2f}%
""")
        
        # Dividend Metrics
        div = data.get('dividend_metrics', {})
        sections.append(f"""
=== DIVIDEND METRICS ===
Dividend Yield: {div.get('dividend_yield', 0):.2f}%
Annual Dividend Rate: ${div.get('dividend_rate', 0):.2f}
Payout Ratio: {div.get('payout_ratio', 0):.2f}%
Dividend Coverage: {div.get('dividend_coverage', 0):.2f}x
FCF Dividend Coverage: {div.get('fcf_dividend_coverage', 0):.2f}x
5-Year Avg Yield: {div.get('five_year_avg_yield', 0):.2f}%
Dividend Sustainability Score: {div.get('dividend_sustainability_score', 0):.0f}/100
""")
        
        # Quality Scores
        qual = data.get('quality_scores', {})
        sections.append(f"""
=== QUALITY SCORES ===
Altman Z-Score: {qual.get('altman_z_score', 0):.2f} ({qual.get('altman_z_interpretation', 'N/A')})
Piotroski F-Score: {qual.get('piotroski_f_score', 0)}/9 ({qual.get('piotroski_f_interpretation', 'N/A')})
DuPont ROE: {qual.get('dupont_roe', 0):.2f}%
  - Net Margin Component: {qual.get('dupont_net_margin', 0):.2f}%
  - Asset Turnover Component: {qual.get('dupont_asset_turnover', 0):.2f}
  - Leverage Component: {qual.get('dupont_equity_multiplier', 0):.2f}x
Quality of Earnings (CFO/NI): {qual.get('quality_of_earnings', 0):.2f}
Accruals Ratio: {qual.get('accruals_ratio', 0):.2f}%
""")
        
        # Historical Valuation
        hist = data.get('historical_valuation', {})
        sections.append(f"""
=== PRICE CONTEXT ===
52-Week High: ${hist.get('fifty_two_week_high', 0):.2f}
52-Week Low: ${hist.get('fifty_two_week_low', 0):.2f}
Distance from 52W High: {hist.get('distance_from_52w_high', 0):.1f}%
52W Range Position: {hist.get('52w_range_position', 0):.1f}%
Price vs 50-Day Avg: {hist.get('price_vs_50d_avg', 0):+.1f}%
Price vs 200-Day Avg: {hist.get('price_vs_200d_avg', 0):+.1f}%
""")
        
        # Peer Comparison
        peer_comp = data.get('peer_comparison', {})
        if peer_comp.get('peers'):
            peer_section = "\n=== PEER COMPARISON ===\n"
            peer_section += f"Peers: {', '.join(peer_comp.get('peers', []))}\n"
            vs_peers = peer_comp.get('vs_peers', {})
            for metric, values in vs_peers.items():
                if values.get('value') and values.get('peer_avg'):
                    peer_section += f"{metric}: {values['value']:.2f} (Peer Avg: {values['peer_avg']:.2f}, Percentile: {values['percentile']:.0f}%)\n"
            sections.append(peer_section)
        
        # Analyst Recommendations
        stats = data.get('key_stats', {})
        if stats.get('recommendation_key'):
            sections.append(f"""
=== ANALYST SENTIMENT ===
Recommendation: {stats.get('recommendation_key', 'N/A')}
Target Price (Mean): ${stats.get('target_mean_price', 0):.2f}
Target Price (High): ${stats.get('target_high_price', 0):.2f}
Target Price (Low): ${stats.get('target_low_price', 0):.2f}
Number of Analysts: {stats.get('number_of_analyst_opinions', 0)}
Beta: {stats.get('beta', 0):.2f}
""")
        
        output = "\n".join(sections)
        
        # Truncate if too long
        if len(output) > max_length:
            output = output[:max_length] + "\n... [truncated]"
        
        return output


# =============================================================================
# Convenience Functions
# =============================================================================

def get_fundamentals(ticker: str) -> Dict:
    """Simple function to get fundamental data"""
    calc = FundamentalCalculator()
    return calc.get_all_fundamentals(ticker)


def get_fundamentals_for_llm(ticker: str) -> str:
    """Simple function to get LLM-formatted fundamental data"""
    calc = FundamentalCalculator()
    return calc.get_formatted_for_llm(ticker)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fundamental Data Fetcher')
    parser.add_argument('ticker', help='Stock ticker (e.g., AAPL)')
    parser.add_argument('--format', '-f', choices=['json', 'llm'], default='llm')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Fetching fundamental data for {args.ticker}")
    print(f"{'='*60}")
    
    calc = FundamentalCalculator(use_cache=not args.no_cache)
    
    if args.format == 'llm':
        print(calc.get_formatted_for_llm(args.ticker))
    else:
        import json as json_module
        data = calc.get_all_fundamentals(args.ticker)
        print(json_module.dumps(data, indent=2, default=str))