"""
XBRL Parser - Extracts structured financial data from SEC filings.

SEC filings include XBRL (eXtensible Business Reporting Language) data
that provides standardized financial metrics. This module fetches and
parses the company facts API which aggregates all XBRL data.
"""

import requests
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import json


@dataclass
class FinancialFact:
    """A single financial fact/metric."""
    label: str
    value: float
    unit: str
    period_end: str
    period_start: Optional[str] = None
    form: str = ""  # 10-K, 10-Q, etc.
    filed: str = ""  # Filing date
    
    @property
    def is_quarterly(self) -> bool:
        """Check if this is a quarterly (vs annual) figure."""
        return self.form == "10-Q"
    
    @property
    def fiscal_year(self) -> int:
        """Extract fiscal year from period end."""
        return int(self.period_end[:4])


@dataclass
class FinancialStatements:
    """Aggregated financial data for a company."""
    ticker: str
    cik: str
    company_name: str
    
    # Income Statement items
    revenue: list[FinancialFact] = field(default_factory=list)
    net_income: list[FinancialFact] = field(default_factory=list)
    gross_profit: list[FinancialFact] = field(default_factory=list)
    operating_income: list[FinancialFact] = field(default_factory=list)
    cost_of_revenue: list[FinancialFact] = field(default_factory=list)
    eps_basic: list[FinancialFact] = field(default_factory=list)
    eps_diluted: list[FinancialFact] = field(default_factory=list)
    
    # Operating expenses
    rd_expense: list[FinancialFact] = field(default_factory=list)
    sga_expense: list[FinancialFact] = field(default_factory=list)
    
    # Balance Sheet items
    total_assets: list[FinancialFact] = field(default_factory=list)
    total_liabilities: list[FinancialFact] = field(default_factory=list)
    stockholders_equity: list[FinancialFact] = field(default_factory=list)
    cash_and_equivalents: list[FinancialFact] = field(default_factory=list)
    total_debt: list[FinancialFact] = field(default_factory=list)
    current_assets: list[FinancialFact] = field(default_factory=list)
    current_liabilities: list[FinancialFact] = field(default_factory=list)
    inventory: list[FinancialFact] = field(default_factory=list)
    accounts_receivable: list[FinancialFact] = field(default_factory=list)
    accounts_payable: list[FinancialFact] = field(default_factory=list)
    goodwill: list[FinancialFact] = field(default_factory=list)
    intangible_assets: list[FinancialFact] = field(default_factory=list)
    
    # Financial sector
    interest_income: list[FinancialFact] = field(default_factory=list)
    interest_expense: list[FinancialFact] = field(default_factory=list)
    
    # Cash Flow items
    operating_cash_flow: list[FinancialFact] = field(default_factory=list)
    capital_expenditures: list[FinancialFact] = field(default_factory=list)
    dividends_paid: list[FinancialFact] = field(default_factory=list)
    
    # Shares
    shares_outstanding: list[FinancialFact] = field(default_factory=list)


class XBRLParser:
    """
    Parses XBRL financial data from SEC EDGAR.
    
    Uses the Company Facts API which provides pre-aggregated XBRL data.
    See: https://www.sec.gov/edgar/sec-api-documentation
    
    Usage:
        parser = XBRLParser("your-email@example.com")
        statements = parser.get_financial_statements("AAPL")
        
        # Get latest annual revenue
        latest_revenue = statements.revenue[0]  # Most recent first
        print(f"Revenue: ${latest_revenue.value:,.0f}")
    """
    
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"
    COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept"
    
    # Mapping of common financial concepts to their XBRL tags
    # Companies may use different tags, so we try multiple
    CONCEPT_MAPPINGS = {
        # Income Statement
        "revenue": [
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
            "us-gaap:Revenues",
            "us-gaap:SalesRevenueNet",
        ],
        "net_income": [
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss",
            "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "gross_profit": [
            "us-gaap:GrossProfit",
        ],
        "operating_income": [
            "us-gaap:OperatingIncomeLoss",
        ],
        "cost_of_revenue": [
            "us-gaap:CostOfGoodsAndServicesSold",
            "us-gaap:CostOfRevenue",
            "us-gaap:CostOfGoodsSold",
        ],
        "eps_basic": [
            "us-gaap:EarningsPerShareBasic",
        ],
        "eps_diluted": [
            "us-gaap:EarningsPerShareDiluted",
        ],
        
        # R&D and SG&A (important for tech/healthcare)
        "rd_expense": [
            "us-gaap:ResearchAndDevelopmentExpense",
            "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],
        "sga_expense": [
            "us-gaap:SellingGeneralAndAdministrativeExpense",
            "us-gaap:GeneralAndAdministrativeExpense",
        ],
        
        # Balance Sheet
        "total_assets": [
            "us-gaap:Assets",
        ],
        "total_liabilities": [
            "us-gaap:Liabilities",
            "us-gaap:LiabilitiesAndStockholdersEquity",
        ],
        "stockholders_equity": [
            "us-gaap:StockholdersEquity",
            "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "cash_and_equivalents": [
            "us-gaap:CashAndCashEquivalentsAtCarryingValue",
            "us-gaap:Cash",
            "us-gaap:CashCashEquivalentsAndShortTermInvestments",
        ],
        "total_debt": [
            "us-gaap:LongTermDebt",
            "us-gaap:LongTermDebtNoncurrent",
            "us-gaap:DebtCurrent",
            "us-gaap:LongTermDebtAndCapitalLeaseObligations",
        ],
        "current_assets": [
            "us-gaap:AssetsCurrent",
        ],
        "current_liabilities": [
            "us-gaap:LiabilitiesCurrent",
        ],
        "inventory": [
            "us-gaap:InventoryNet",
            "us-gaap:InventoryFinishedGoods",
        ],
        "accounts_receivable": [
            "us-gaap:AccountsReceivableNetCurrent",
            "us-gaap:AccountsReceivableNet",
        ],
        "accounts_payable": [
            "us-gaap:AccountsPayableCurrent",
            "us-gaap:AccountsPayable",
        ],
        "goodwill": [
            "us-gaap:Goodwill",
        ],
        "intangible_assets": [
            "us-gaap:IntangibleAssetsNetExcludingGoodwill",
            "us-gaap:FiniteLivedIntangibleAssetsNet",
        ],
        
        # Financial sector specific
        "interest_income": [
            "us-gaap:InterestIncomeExpenseNet",
            "us-gaap:InterestAndDividendIncomeOperating",
        ],
        "interest_expense": [
            "us-gaap:InterestExpense",
            "us-gaap:InterestExpenseDebt",
        ],
        
        # Cash Flow
        "operating_cash_flow": [
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        ],
        "capital_expenditures": [
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            "us-gaap:CapitalExpendituresIncurredButNotYetPaid",
        ],
        "dividends_paid": [
            "us-gaap:PaymentsOfDividends",
            "us-gaap:PaymentsOfDividendsCommonStock",
        ],
        
        # Shares
        "shares_outstanding": [
            "us-gaap:CommonStockSharesOutstanding",
            "dei:EntityCommonStockSharesOutstanding",
        ],
    }
    
    REQUEST_DELAY = 0.1
    
    def __init__(self, user_email: str):
        """
        Initialize parser with contact email (required by SEC).
        
        Args:
            user_email: Your email for SEC User-Agent
        """
        self.user_email = user_email
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"InvestmentAgent/1.0 ({user_email})",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json"
        })
        self._last_request_time = 0
        self._ticker_to_cik: dict[str, str] = {}
    
    def _rate_limit(self):
        """Enforce SEC rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, url: str) -> Optional[dict]:
        """Make rate-limited GET request."""
        self._rate_limit()
        response = self.session.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    
    def _load_ticker_map(self):
        """Load ticker-to-CIK mapping."""
        if self._ticker_to_cik:
            return
        
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(url)
        
        if data is None:
            raise RuntimeError("Failed to load ticker-to-CIK mapping from SEC")
        
        for entry in data.values():
            ticker = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            self._ticker_to_cik[ticker] = cik
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a ticker symbol."""
        self._load_ticker_map()
        return self._ticker_to_cik.get(ticker.upper())
    
    def _extract_facts(
        self, 
        facts_data: dict, 
        concept_tags: list[str],
        annual_only: bool = False
    ) -> list[FinancialFact]:
        """
        Extract facts for a given concept from company facts data.
        
        Args:
            facts_data: Raw company facts JSON
            concept_tags: List of XBRL tags to try
            annual_only: If True, only return 10-K data
            
        Returns:
            List of FinancialFact, sorted by period_end descending
        """
        us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
        dei = facts_data.get("facts", {}).get("dei", {})
        
        best_facts = []
        best_max_date = ""
        
        for tag in concept_tags:
            # Parse namespace and concept name
            if ":" in tag:
                namespace, concept = tag.split(":")
            else:
                namespace, concept = "us-gaap", tag
            
            source = us_gaap if namespace == "us-gaap" else dei
            concept_data = source.get(concept, {})
            
            if not concept_data:
                continue
            
            # Get the units (usually USD or shares)
            units = concept_data.get("units", {})
            
            facts = []
            for unit_type, entries in units.items():
                for entry in entries:
                    # Skip if we only want annual and this isn't 10-K
                    if annual_only and entry.get("form") != "10-K":
                        continue
                    
                    # For 10-K filings, try to identify full-year figures
                    # by checking if the period spans approximately a year
                    start = entry.get("start")
                    end = entry.get("end", "")
                    form = entry.get("form", "")
                    
                    # If this is a 10-K and we have both start and end,
                    # verify it's a full year period (not a quarter)
                    if form == "10-K" and start and end:
                        try:
                            from datetime import datetime
                            start_dt = datetime.strptime(start, "%Y-%m-%d")
                            end_dt = datetime.strptime(end, "%Y-%m-%d")
                            days = (end_dt - start_dt).days
                            # Full year is ~365 days, skip if < 300 (probably quarterly)
                            if days < 300:
                                continue
                        except ValueError:
                            pass
                    
                    fact = FinancialFact(
                        label=concept_data.get("label", concept),
                        value=entry.get("val", 0),
                        unit=unit_type,
                        period_end=end,
                        period_start=start,
                        form=form,
                        filed=entry.get("filed", "")
                    )
                    facts.append(fact)
            
            if facts:
                # Sort by period end date, most recent first
                facts.sort(key=lambda x: x.period_end, reverse=True)
                
                # Deduplicate by period_end + form (keep most recent filing)
                seen = set()
                unique_facts = []
                for f in facts:
                    key = (f.period_end, f.form)
                    if key not in seen:
                        seen.add(key)
                        unique_facts.append(f)
                
                # Track which tag has the most recent data
                if unique_facts:
                    max_date = unique_facts[0].period_end
                    if max_date > best_max_date:
                        best_max_date = max_date
                        best_facts = unique_facts
        
        return best_facts
    
    def get_company_facts(self, ticker: str) -> Optional[dict]:
        """
        Get all XBRL facts for a company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Raw company facts JSON or None
        """
        cik = self.get_cik(ticker)
        if not cik:
            return None
        
        url = f"{self.COMPANY_FACTS_URL}/CIK{cik}.json"
        return self._get(url)
    
    def get_financial_statements(self, ticker: str) -> Optional[FinancialStatements]:
        """
        Get structured financial statements for a company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            FinancialStatements object with all available data
        """
        facts_data = self.get_company_facts(ticker)
        if not facts_data:
            return None
        
        cik = self.get_cik(ticker)
        if cik is None:
            return None
        
        company_name = facts_data.get("entityName", "")
        
        statements = FinancialStatements(
            ticker=ticker.upper(),
            cik=cik,
            company_name=company_name
        )
        
        # Extract each financial metric
        for metric_name, concept_tags in self.CONCEPT_MAPPINGS.items():
            facts = self._extract_facts(facts_data, concept_tags)
            setattr(statements, metric_name, facts)
        
        return statements
    
    def get_latest_metrics(self, ticker: str, annual_only: bool = True) -> dict[str, Any]:
        """
        Get the most recent value for each key metric.
        
        Args:
            ticker: Stock ticker symbol
            annual_only: If True, only use 10-K data
            
        Returns:
            Dict with metric names and their latest values
        """
        statements = self.get_financial_statements(ticker)
        if not statements:
            return {}
        
        metrics = {}
        
        for metric_name in self.CONCEPT_MAPPINGS.keys():
            facts = getattr(statements, metric_name, [])
            
            if annual_only:
                facts = [f for f in facts if f.form == "10-K"]
            
            if facts:
                latest = facts[0]
                metrics[metric_name] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "period_end": latest.period_end,
                    "form": latest.form
                }
        
        return metrics


# --- Demo / Testing ---

if __name__ == "__main__":
    parser = XBRLParser("demo@example.com")
    
    print("=== Fetching AAPL Financial Data ===\n")
    
    statements = parser.get_financial_statements("AAPL")
    
    if statements:
        print(f"Company: {statements.company_name}")
        print(f"CIK: {statements.cik}\n")
        
        # Show recent revenue
        print("=== Revenue (Last 5 Years) ===")
        annual_revenue = [f for f in statements.revenue if f.form == "10-K"][:5]
        for rev in annual_revenue:
            print(f"  {rev.period_end}: ${rev.value/1e9:,.1f}B")
        
        # Show recent net income
        print("\n=== Net Income (Last 5 Years) ===")
        annual_ni = [f for f in statements.net_income if f.form == "10-K"][:5]
        for ni in annual_ni:
            print(f"  {ni.period_end}: ${ni.value/1e9:,.1f}B")
        
        # Show balance sheet snapshot
        print("\n=== Latest Balance Sheet ===")
        if statements.total_assets:
            latest = statements.total_assets[0]
            print(f"  Total Assets: ${latest.value/1e9:,.1f}B ({latest.period_end})")
        if statements.stockholders_equity:
            latest = statements.stockholders_equity[0]
            print(f"  Stockholders Equity: ${latest.value/1e9:,.1f}B")
        if statements.cash_and_equivalents:
            latest = statements.cash_and_equivalents[0]
            print(f"  Cash: ${latest.value/1e9:,.1f}B")
    
    print("\n=== Latest Metrics Summary ===")
    metrics = parser.get_latest_metrics("AAPL", annual_only=True)
    for name, data in metrics.items():
        if data["unit"] == "USD":
            print(f"  {name}: ${data['value']/1e9:,.2f}B")
        elif data["unit"] == "USD/shares":
            print(f"  {name}: ${data['value']:.2f}")
        else:
            print(f"  {name}: {data['value']:,.0f} {data['unit']}")