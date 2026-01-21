"""
EDGAR Fetcher - Retrieves SEC filings from the EDGAR database.

SEC EDGAR requires a User-Agent header with contact info.
See: https://www.sec.gov/os/accessing-edgar-data
"""

import requests
import time
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class CompanyInfo:
    """Basic company information from EDGAR."""
    cik: str
    name: str
    ticker: str
    sic: str  # Standard Industrial Classification
    sic_description: str
    fiscal_year_end: str


@dataclass
class Filing:
    """Represents a single SEC filing."""
    accession_number: str
    form_type: str
    filing_date: str
    report_date: str
    primary_document: str
    description: str
    
    @property
    def url(self) -> str:
        """Construct the URL to the filing document."""
        acc_no_clean = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{self._cik}/{acc_no_clean}/{self.primary_document}"
    
    def set_cik(self, cik: str):
        """Set CIK for URL construction."""
        self._cik = cik


class EdgarFetcher:
    """
    Fetches SEC filings from EDGAR.
    
    Usage:
        fetcher = EdgarFetcher("your-email@example.com")
        filings = fetcher.get_recent_filings("AAPL", form_types=["10-K", "10-Q"])
    """
    
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/submissions"
    
    # Rate limit: SEC requires max 10 requests/second
    REQUEST_DELAY = 0.1
    
    def __init__(self, user_email: str):
        """
        Initialize the fetcher with contact email (required by SEC).
        
        Args:
            user_email: Your email address for SEC User-Agent requirement
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
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, url: str) -> dict:
        """Make a rate-limited GET request."""
        self._rate_limit()
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def _load_ticker_map(self):
        """Load the ticker-to-CIK mapping from SEC."""
        if self._ticker_to_cik:
            return
        
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(url)
        
        # Build ticker -> CIK map (CIK needs to be zero-padded to 10 digits)
        for entry in data.values():
            ticker = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            self._ticker_to_cik[ticker] = cik
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get the CIK (Central Index Key) for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            10-digit CIK string or None if not found
        """
        self._load_ticker_map()
        return self._ticker_to_cik.get(ticker.upper())
    
    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """
        Get company information from EDGAR.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CompanyInfo object or None if not found
        """
        cik = self.get_cik(ticker)
        if not cik:
            return None
        
        url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
        data = self._get(url)
        
        return CompanyInfo(
            cik=cik,
            name=data.get("name", ""),
            ticker=ticker.upper(),
            sic=data.get("sic", ""),
            sic_description=data.get("sicDescription", ""),
            fiscal_year_end=data.get("fiscalYearEnd", "")
        )
    
    def get_recent_filings(
        self, 
        ticker: str, 
        form_types: Optional[list[str]] = None,
        limit: int = 20
    ) -> list[Filing]:
        """
        Get recent filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            form_types: Filter by form types (e.g., ["10-K", "10-Q", "8-K"])
            limit: Maximum number of filings to return
            
        Returns:
            List of Filing objects, most recent first
        """
        cik = self.get_cik(ticker)
        if not cik:
            raise ValueError(f"Unknown ticker: {ticker}")
        
        url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
        data = self._get(url)
        
        recent = data.get("filings", {}).get("recent", {})
        
        filings = []
        for i in range(len(recent.get("accessionNumber", []))):
            form = recent["form"][i]
            
            # Filter by form type if specified
            if form_types and form not in form_types:
                continue
            
            filing = Filing(
                accession_number=recent["accessionNumber"][i],
                form_type=form,
                filing_date=recent["filingDate"][i],
                report_date=recent.get("reportDate", [""])[i] or recent["filingDate"][i],
                primary_document=recent["primaryDocument"][i],
                description=recent.get("primaryDocDescription", [""])[i]
            )
            filing.set_cik(cik)
            filings.append(filing)
            
            if len(filings) >= limit:
                break
        
        return filings
    
    def get_filing_document(self, filing: Filing, cik: str) -> str:
        """
        Fetch the raw document content for a filing.
        
        Args:
            filing: Filing object
            cik: Company CIK (10-digit)
            
        Returns:
            Raw document content (HTML or XML)
        """
        filing.set_cik(cik)
        self._rate_limit()
        
        response = self.session.get(filing.url)
        response.raise_for_status()
        return response.text


# --- Demo / Testing ---

if __name__ == "__main__":
    # Example usage
    fetcher = EdgarFetcher("iswagnacio@gmail.com")
    
    # Get company info
    print("=== Company Info ===")
    info = fetcher.get_company_info("AAPL")
    if info:
        print(f"Name: {info.name}")
        print(f"CIK: {info.cik}")
        print(f"SIC: {info.sic} - {info.sic_description}")
        print(f"Fiscal Year End: {info.fiscal_year_end}")
    
    print("\n=== Recent 10-K and 10-Q Filings ===")
    filings = fetcher.get_recent_filings("AAPL", form_types=["10-K", "10-Q"], limit=5)
    for f in filings:
        print(f"{f.filing_date} | {f.form_type:5} | {f.description[:50]}")