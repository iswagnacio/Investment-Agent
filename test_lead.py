"""
Test script for General Analyst Agent
Run: python test_general_analyst.py

This tests:
1. Data gathering from all sources
2. Score calculations
3. Quick recommendations
4. Full LLM-powered analysis (requires ANTHROPIC_API_KEY)
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("GENERAL ANALYST AGENT TEST")
print("=" * 70)

# Check for API key
has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
print(f"\nANTHROPIC_API_KEY: {'✓ Set' if has_api_key else '✗ Not set'}")

if not has_api_key:
    print("\nWarning: ANTHROPIC_API_KEY not set. LLM analysis will not be available.")
    print("Set the key in .env file to enable full analysis.")

# Test imports
print("\n--- Testing Imports ---")
try:
    from lead_agent import (
        GeneralAnalystAgent, 
        ScoreCalculator,
        Signal,
        OptionsStrategy
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test Score Calculator (no API key needed)
print("\n" + "=" * 70)
print("TEST 1: Score Calculator")
print("=" * 70)

scorer = ScoreCalculator()

# Test technical score calculation
mock_technical = {
    'price': {'current': 150.0, 'change_1d': 1.5, 'change_5d': 3.0, 'change_20d': 8.0},
    'moving_averages': {'sma_20': 145.0, 'sma_50': 140.0, 'sma_200': 130.0, 'ema_12': 148.0, 'ema_26': 146.0},
    'momentum': {'rsi_14': 55.0, 'momentum_10d': 5.0},
    'trend': {'macd': {'macd': 2.5, 'signal': 2.0, 'histogram': 0.5}, 'adx': 28.0},
    'volatility': {'bollinger': {'upper': 160.0, 'middle': 145.0, 'lower': 130.0, 'width': 30.0}, 
                   'atr': 3.5, 'historical_vol': 0.25},
    'volume': {'current': 50000000, 'average': 45000000, 'ratio': 1.1, 'trend': 0.05}
}

tech_score, tech_breakdown = scorer.calculate_technical_score(mock_technical)
print(f"\nTechnical Score: {tech_score}/100")
print(f"Breakdown: {json.dumps(tech_breakdown, indent=2)}")

# Test signal conversion
print(f"\nSignal Tests:")
for score in [80, 65, 50, 35, 20]:
    signal = scorer.score_to_signal(score)
    print(f"  Score {score} -> {signal.value}")

# Test options strategy
print(f"\nOptions Strategy Tests:")
for signal, vol in [(Signal.STRONG_BUY, 0.2), (Signal.BUY, 0.5), (Signal.HOLD, 0.3), (Signal.SELL, 0.4)]:
    strategy, rationale = scorer.determine_options_strategy(signal, vol)
    print(f"  {signal.value} + Vol {vol:.1f} -> {strategy.value}")

# Test with real API if available
if has_api_key:
    print("\n" + "=" * 70)
    print("TEST 2: Quick Recommendation (Real Data)")
    print("=" * 70)
    
    agent = GeneralAnalystAgent()
    
    test_ticker = "AAPL"
    print(f"\nGetting quick recommendation for {test_ticker}...")
    
    try:
        rec = agent.get_quick_recommendation(test_ticker)
        
        print(f"\nResults for {rec['ticker']} ({rec['company']}):")
        print(f"  Current Price: ${rec['current_price']:.2f}")
        print(f"\n  Scores:")
        print(f"    Technical:    {rec['scores']['technical']:.1f}/100")
        print(f"    Fundamental:  {rec['scores']['fundamental']:.1f}/100")
        print(f"    Sentiment:    {rec['scores']['sentiment']:.1f}/100")
        print(f"    Overall:      {rec['scores']['overall']:.1f}/100")
        print(f"\n  Signal: {rec['signal']}")
        print(f"  Confidence: {rec['confidence']}%")
        print(f"\n  Price Targets ({rec['price_targets']['timeframe']}):")
        print(f"    Low:  ${rec['price_targets']['low']:.2f}")
        print(f"    Mid:  ${rec['price_targets']['mid']:.2f}")
        print(f"    High: ${rec['price_targets']['high']:.2f}")
        print(f"\n  Options Strategy: {rec['options']['strategy']}")
        print(f"  Rationale: {rec['options']['rationale']}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("TEST 3: Stock Comparison")
    print("=" * 70)
    
    compare_tickers = ["AAPL", "MSFT", "GOOGL"]
    print(f"\nComparing: {', '.join(compare_tickers)}")
    
    try:
        comparison = agent.compare_stocks(compare_tickers)
        print(comparison)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("TEST 4: Full Analysis (LLM-Powered)")
    print("=" * 70)
    
    print("\nNote: Full analysis uses LLM API credits.")
    print("Skipping by default. To run full analysis:")
    print("  agent.analyze('AAPL')")
    
    # Uncomment to run full analysis:
    # print("\nRunning full analysis for AAPL...")
    # try:
    #     analysis = agent.analyze("AAPL")
    #     print(analysis)
    # except Exception as e:
    #     print(f"Error: {e}")

else:
    print("\n" + "=" * 70)
    print("SKIPPING LIVE TESTS (No API Key)")
    print("=" * 70)
    print("\nTo run live tests, set ANTHROPIC_API_KEY in your .env file")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")