#!/usr/bin/env python3
"""
Demo script - Test the EDGAR fetcher and fundamental analyzer.

Run this to verify the setup works:
    python demo.py AAPL
    python demo.py MSFT 425.00
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from edgar_fetch import EdgarFetcher
from xbrl_parse import XBRLParser
from analyzer import FundamentalAnalyzer


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python demo.py <TICKER> [PRICE]")
        print("Example: python demo.py AAPL 175.50")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    price = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Use a placeholder email - in production, use your real email
    email = "demo@example.com"
    
    print(f"{'='*60}")
    print(f"  Investment Agent - Fundamental Analysis Demo")
    print(f"  Ticker: {ticker}")
    if price:
        print(f"  Price: ${price:.2f}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch company info from EDGAR
    print("[1/4] Fetching company info from SEC EDGAR...")
    fetcher = EdgarFetcher(email)
    
    company_info = fetcher.get_company_info(ticker)
    if not company_info:
        print(f"❌ Could not find company with ticker: {ticker}")
        sys.exit(1)
    
    print(f"  ✓ Found: {company_info.name}")
    print(f"    CIK: {company_info.cik}")
    print(f"    Industry: {company_info.sic_description}")
    print()
    
    # Step 2: Get recent filings
    print("[2/4] Fetching recent SEC filings...")
    filings = fetcher.get_recent_filings(ticker, form_types=["10-K", "10-Q"], limit=5)
    
    print(f"  ✓ Found {len(filings)} recent filings:")
    for f in filings[:5]:
        print(f"    {f.filing_date} | {f.form_type:5} | {f.description[:40]}...")
    print()
    
    # Step 3: Parse XBRL financial data
    print("[3/4] Parsing XBRL financial data...")
    parser = XBRLParser(email)
    statements = parser.get_financial_statements(ticker)
    
    if statements:
        print("  ✓ Successfully parsed financial statements")
        
        # Show a preview of available data - filter for full year periods only
        annual_revenue = [f for f in statements.revenue 
                         if f.form == "10-K" and f.period_start 
                         and (int(f.period_end[:4]) - int(f.period_start[:4]) >= 0)]
        # Deduplicate by fiscal year
        seen_years = set()
        unique_annual = []
        for rev in annual_revenue:
            year = rev.period_end[:4]
            if year not in seen_years:
                seen_years.add(year)
                unique_annual.append(rev)
        
        if unique_annual:
            print(f"\n  Revenue history (annual):")
            for rev in unique_annual[:3]:
                print(f"    FY{rev.period_end[:4]}: ${rev.value/1e9:,.1f}B")
    else:
        print("  ⚠️ Could not parse XBRL data")
    print()
    
    # Step 4: Run fundamental analysis
    print("[4/4] Running fundamental analysis...")
    analyzer = FundamentalAnalyzer(email)
    analysis = analyzer.analyze(ticker, current_price=price)
    
    if analysis:
        print("  ✓ Analysis complete\n")
        print(analyzer.generate_summary(analysis))
    else:
        print("  ❌ Analysis failed")
    
    print(f"\n{'='*60}")
    print("  Demo complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()