# technical_analyzer_agent.py

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from typing import Dict, List
import json
import os

from technical import TechnicalCalculator


class TechnicalAnalyzerAgent:
    """
    Uses LLM to intelligently select relevant technical indicators
    based on stock characteristics (sector, industry, size).
    """
    
    def __init__(self, model: str = "claude-4-5-sonnet-20241022"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.calculator = TechnicalCalculator()
        self.llm = ChatAnthropic(model_name=model, temperature=0)
    
    def analyze(self, ticker: str) -> str:
        """
        Perform intelligent technical analysis:
        1. Get stock metadata (sector, industry, market cap)
        2. Calculate all technical indicators
        3. Ask LLM to select most relevant indicators
        4. Generate contextual analysis report
        """
        
        # Step 1: Get stock info
        stock_info = self.calculator.get_stock_info(ticker)
        
        # Step 2: Calculate all indicators
        indicators = self.calculator.get_all_indicators(ticker)
        
        # Step 3: Ask LLM to select relevant indicators and analyze
        prompt = self._create_analysis_prompt()
        
        response = self.llm.invoke(prompt.format(
            ticker=ticker.upper(),
            stock_name=stock_info['name'],
            sector=stock_info['sector'],
            industry=stock_info['industry'],
            market_cap=stock_info['market_cap'],
            indicators=json.dumps(indicators, indent=2, default=str)
        ))
        
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            # Handle case where content is a list (e.g., with tool calls)
            text_parts = [item for item in response.content if isinstance(item, str)]
            return '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            return str(response.content)
    
    def _create_analysis_prompt(self) -> PromptTemplate:
        """Create prompt for LLM to intelligently analyze technical indicators"""
        
        template = """You are a technical analysis expert. Analyze the following stock and select the MOST RELEVANT technical indicators based on its characteristics.

Stock: {ticker} - {stock_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

Available Technical Indicators:
{indicators}

Your task:
1. Determine the stock's category:
   - Large-cap (>$200B): Focus on trend-following, moving averages, institutional volume
   - Mid-cap ($10B-$200B): Balance of trend and momentum indicators
   - Small-cap (<$10B): Focus on volatility, momentum, and volume patterns
   
2. Consider sector characteristics:
   - Tech: Fast-moving, use shorter timeframes (RSI, MACD, EMA)
   - Financials: Sensitive to rates, use trend indicators (ADX, moving averages)
   - Utilities: Slow-moving, focus on longer-term trends (SMA 50/200)
   - Healthcare/Biotech: High volatility, use Bollinger Bands, ATR
   - Consumer: Seasonal patterns, volume analysis important
   - Energy: Volatile, momentum and volatility indicators

3. Select the 4-6 MOST RELEVANT indicators for this specific stock.

4. Provide a focused analysis using only those selected indicators. Explain:
   - Why you chose these specific indicators
   - What each indicator is currently showing
   - How they relate to this stock's sector/size characteristics
   - Any notable patterns or divergences

DO NOT provide buy/sell recommendations. Focus on objective technical observations.

Keep the analysis concise and actionable - about 200-300 words."""

        return PromptTemplate.from_template(template)
    
    def batch_analyze(self, tickers: List[str]) -> Dict[str, str]:
        """Analyze multiple tickers"""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.analyze(ticker)
            except Exception as e:
                results[ticker] = f"Error analyzing {ticker}: {str(e)}"
        return results


# Example usage
if __name__ == "__main__":
    agent = TechnicalAnalyzerAgent()
    
    print("=== Single Stock Analysis ===")
    print(agent.analyze("AAPL"))
    
    print("\n" + "="*80 + "\n")
    
    print("=== Batch Analysis (Different Sectors/Sizes) ===")
    tickers = ["AAPL", "JPM", "TSLA"]  # Tech large-cap, Financial, Volatile growth
    results = agent.batch_analyze(tickers)
    
    for ticker, analysis in results.items():
        print(f"\n{ticker}:")
        print("-" * 80)
        print(analysis)
        print()