"""
Test script for Fundamental Analysis modules.
Run: python test_fundamental.py

This tests:
1. FundamentalCalculator - Data fetching and metric calculation
2. FundamentalAnalystAgent - LLM-powered analysis (requires ANTHROPIC_API_KEY)
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Test 1: Data Fetcher
print("=" * 60)
print("TEST 1: Fundamental Calculator (Data Fetching)")
print("=" * 60)

from fundamental import FundamentalCalculator

calc = FundamentalCalculator(use_cache=True)

# Test company info
print("\n--- Company Info ---")
info = calc.get_company_info("AAPL")
print(f"Name: {info.get('name')}")
print(f"Sector: {info.get('sector')}")
print(f"Industry: {info.get('industry')}")
print(f"Market Cap: ${info.get('market_cap', 0):,.0f}")

# Test full fundamentals
print("\n--- Fetching Full Fundamentals (this may take a moment) ---")
data = calc.get_all_fundamentals("AAPL")

print("\nValuation Ratios:")
val = data.get('valuation_ratios', {})
print(f"  P/E Ratio: {val.get('pe_ratio', 0):.2f}")
print(f"  P/B Ratio: {val.get('price_to_book', 0):.2f}")
print(f"  PEG Ratio: {val.get('peg_ratio', 0):.2f}")
print(f"  EV/EBITDA: {val.get('ev_to_ebitda', 0):.2f}")

print("\nProfitability Ratios:")
prof = data.get('profitability_ratios', {})
print(f"  Gross Margin: {prof.get('gross_margin', 0):.2f}%")
print(f"  Net Margin: {prof.get('net_profit_margin', 0):.2f}%")
print(f"  ROE: {prof.get('return_on_equity', 0):.2f}%")
print(f"  ROIC: {prof.get('return_on_invested_capital', 0):.2f}%")

print("\nQuality Scores:")
qual = data.get('quality_scores', {})
print(f"  Altman Z-Score: {qual.get('altman_z_score', 0):.2f} ({qual.get('altman_z_interpretation', 'N/A')})")
print(f"  Piotroski F-Score: {qual.get('piotroski_f_score', 0)}/9 ({qual.get('piotroski_f_interpretation', 'N/A')})")
print(f"  DuPont ROE: {qual.get('dupont_roe', 0):.2f}%")

print("\nGrowth Metrics:")
growth = data.get('growth_metrics', {})
print(f"  Revenue Growth YoY: {growth.get('revenue_growth_yoy', 0):.2f}%")
print(f"  Earnings Growth YoY: {growth.get('earnings_growth_yoy', 0):.2f}%")
print(f"  EBITDA Growth YoY: {growth.get('ebitda_growth_yoy', 0):.2f}%")

print("\nDividend Metrics:")
div = data.get('dividend_metrics', {})
print(f"  Dividend Yield: {div.get('dividend_yield', 0):.2f}%")
print(f"  Payout Ratio: {div.get('payout_ratio', 0):.2f}%")
print(f"  Dividend Coverage: {div.get('dividend_coverage', 0):.2f}x")
print(f"  Sustainability Score: {div.get('dividend_sustainability_score', 0):.0f}/100")

print("\nLeverage Ratios:")
lev = data.get('leverage_ratios', {})
print(f"  Debt/Equity: {lev.get('debt_to_equity', 0):.2f}")
print(f"  Interest Coverage: {lev.get('interest_coverage', 0):.2f}x")

print("\nLiquidity Ratios:")
liq = data.get('liquidity_ratios', {})
print(f"  Current Ratio: {liq.get('current_ratio', 0):.2f}")
print(f"  Quick Ratio: {liq.get('quick_ratio', 0):.2f}")

# Test LLM-formatted output
print("\n--- LLM Formatted Output (preview) ---")
llm_formatted = calc.get_formatted_for_llm("AAPL")
# Print first 1500 chars
print(llm_formatted[:1500] + "\n... [truncated]")

# Test 2: Fundamental Agent (requires API key)
print("\n" + "=" * 60)
print("TEST 2: Fundamental Analyst Agent (LLM-Powered)")
print("=" * 60)

if os.getenv("ANTHROPIC_API_KEY"):
    from fundamental_agent import FundamentalAnalystAgent
    
    agent = FundamentalAnalystAgent()
    
    # Quick screen
    print("\n--- Quick Screen ---")
    screen = agent.quick_screen("AAPL")
    print(screen)
    
    # Note: Full analysis takes longer and uses API credits
    print("\n--- Comprehensive Analysis ---")
    print("(Skipping full LLM analysis to save API credits)")
    print("To run full analysis, use:")
    print("  agent.analyze('AAPL', 'comprehensive')")
    print("  agent.analyze('AAPL', 'valuation')")
    print("  agent.analyze('AAPL', 'quality')")
    print("  agent.analyze('AAPL', 'dividend')")
    print("  agent.analyze('AAPL', 'growth')")
    print("  agent.compare_stocks(['AAPL', 'MSFT', 'GOOGL'])")
else:
    print("\nANTHROPIC_API_KEY not set - skipping LLM agent tests")
    print("Set the key in .env file to enable LLM-powered analysis")

print("\n" + "=" * 60)
print("Tests completed!")
print("=" * 60)