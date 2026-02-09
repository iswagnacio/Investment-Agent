"""
Technical Analyzer Agent Module
================================
Uses LLM to intelligently select relevant technical indicators
based on stock characteristics (sector, industry, size).

UPDATED: Dual-mode output (mirrors fundamental_agent pattern)
- "for_synthesis": Returns structured JSON signals for lead agent consumption
- "comprehensive": Returns detailed prose analysis (fetched on demand)

Usage:
    from technical_agent import TechnicalAnalyzerAgent
    
    agent = TechnicalAnalyzerAgent()
    
    # Structured signals (default for lead agent pipeline)
    signals = agent.analyze("AAPL", analysis_type="for_synthesis")
    
    # Detailed prose (on demand)
    report = agent.analyze("AAPL", analysis_type="comprehensive")
"""

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional
import json
from dotenv import load_dotenv
import os

from technical import TechnicalCalculator

load_dotenv()


class TechnicalAnalyzerAgent:
    """
    Uses LLM to intelligently select relevant technical indicators
    based on stock characteristics (sector, industry, size).
    
    Supports two output modes:
    - "for_synthesis": Structured JSON for lead agent integration
    - "comprehensive": Detailed prose analysis
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.calculator = TechnicalCalculator()
        self.llm = ChatAnthropic(model_name=model, temperature=0, max_tokens=4096)
    
    def analyze(self, ticker: str, analysis_type: str = "comprehensive") -> str:
        """
        Perform intelligent technical analysis.
        
        Args:
            ticker: Stock ticker symbol
            analysis_type: Type of analysis
                - "comprehensive": Full prose analysis (original behavior)
                - "for_synthesis": Structured JSON output for lead agent
        
        Returns:
            Analysis string (prose or JSON depending on mode)
        """
        ticker = ticker.upper()
        
        # Step 1: Get stock info
        stock_info = self.calculator.get_stock_info(ticker)
        
        # Step 2: Calculate all indicators
        indicators = self.calculator.get_all_indicators(ticker)
        
        # Step 3: Select prompt based on analysis type
        if analysis_type == "for_synthesis":
            prompt = self._create_synthesis_prompt()
        else:
            prompt = self._create_comprehensive_prompt()
        
        response = self.llm.invoke(prompt.format(
            ticker=ticker,
            stock_name=stock_info['name'],
            sector=stock_info['sector'],
            industry=stock_info['industry'],
            market_cap=stock_info['market_cap'],
            indicators=json.dumps(indicators, indent=2, default=str)
        ))
        
        if isinstance(response.content, str):
            result = response.content
        elif isinstance(response.content, list):
            text_parts = [item for item in response.content if isinstance(item, str)]
            result = '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            result = str(response.content)
        
        # Add header for comprehensive mode (mirrors fundamental_agent)
        if analysis_type == "comprehensive":
            header = f"=== TECHNICAL ANALYSIS FOR {ticker} ({stock_info['name']}) ===\n\n"
            return header + result
        
        return result
    
    def _create_synthesis_prompt(self) -> PromptTemplate:
        """
        Create prompt for structured JSON output (for lead agent integration).
        Mirrors fundamental_agent's for_synthesis pattern.
        """
        
        template = """You are a technical analyst providing STRUCTURED analysis for algorithmic integration.

**ANALYZING: {ticker} ({stock_name})**
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

Available Technical Indicators:
{indicators}

=== YOUR TASK ===

First, determine which indicators are MOST RELEVANT for this stock based on:
- Sector: {sector} (e.g., Tech = shorter timeframes; Utilities = longer trends)
- Size: Market Cap ${market_cap:,} (Large-cap = trend-following; Small-cap = momentum/volatility)
- Industry: {industry}

Then analyze {ticker} and provide a STRUCTURED JSON response (not prose).

Return ONLY valid JSON in this exact format:

{{
  "ticker": "{ticker}",
  "stock_name": "{stock_name}",
  "selected_indicators": ["indicator1", "indicator2", "indicator3", "indicator4"],
  "selection_reasoning": "One sentence on why these indicators are most relevant for {ticker}",
  "llm_analysis": {{
    "trend_verdict": {{
      "signal": "Bullish|Bearish|Neutral",
      "strength": "Strong|Moderate|Weak",
      "confidence": "High|Medium|Low",
      "reasoning": "One sentence about {ticker}'s trend using selected indicators"
    }},
    "momentum_verdict": {{
      "signal": "Bullish|Bearish|Neutral|Oversold|Overbought",
      "confidence": "High|Medium|Low",
      "reasoning": "One sentence about {ticker}'s momentum"
    }},
    "volatility_verdict": {{
      "signal": "Low|Normal|High|Extreme",
      "implication": "One sentence on what volatility means for {ticker} positioning",
      "bollinger_position": "Above Upper|Upper Half|Middle|Lower Half|Below Lower"
    }},
    "volume_verdict": {{
      "signal": "Accumulation|Distribution|Normal|Low Interest",
      "reasoning": "One sentence about {ticker}'s volume pattern"
    }},
    "key_levels": {{
      "support_1": 0.00,
      "support_2": 0.00,
      "resistance_1": 0.00,
      "resistance_2": 0.00,
      "derivation": "Brief note on how levels were derived (e.g., SMA, Bollinger, recent price action)"
    }},
    "pattern_detected": "Short description of dominant pattern (e.g., 'Consolidation above SMA(50)', 'Breakout with volume confirmation', 'Bearish divergence RSI vs price')",
    "near_term_outlook": {{
      "bias": "Bullish|Bearish|Neutral",
      "timeframe": "1-2 weeks|2-4 weeks|1-3 months",
      "catalyst_level": "Price above/below $XX would confirm direction"
    }},
    "risk_flags": ["risk 1 for {ticker}", "risk 2 for {ticker}"]
  }}
}}

**CRITICAL RULES:**
- All analysis must be about {ticker} ({stock_name}). Do NOT reference other stocks.
- Support/resistance levels must be derived from the actual indicator data provided.
- Return ONLY valid JSON, no preamble or markdown fences.
- selected_indicators should list 4-6 indicators you consider most relevant for this stock's profile."""

        return PromptTemplate.from_template(template)
    
    def _create_comprehensive_prompt(self) -> PromptTemplate:
        """
        Create prompt for detailed prose analysis.
        Enhanced from original with stronger ticker anchoring.
        """
        
        template = """You are a technical analysis expert. Analyze the following stock and select the MOST RELEVANT technical indicators based on its characteristics.

**CRITICAL: You are analyzing {ticker} ({stock_name}). Every statement must be about THIS stock only.**

Stock: {ticker} - {stock_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

Available Technical Indicators:
{indicators}

Your task:
1. Determine {ticker}'s category:
   - Large-cap (>$200B): Focus on trend-following, moving averages, institutional volume
   - Mid-cap ($10B-$200B): Balance of trend and momentum indicators
   - Small-cap (<$10B): Focus on volatility, momentum, and volume patterns
   
2. Consider {ticker}'s sector characteristics:
   - Tech: Fast-moving, use shorter timeframes (RSI, MACD, EMA)
   - Financials: Sensitive to rates, use trend indicators (ADX, moving averages)
   - Utilities: Slow-moving, focus on longer-term trends (SMA 50/200)
   - Healthcare/Biotech: High volatility, use Bollinger Bands, ATR
   - Consumer: Seasonal patterns, volume analysis important
   - Energy: Volatile, momentum and volatility indicators

3. Select the 4-6 MOST RELEVANT indicators for {ticker} specifically.

4. Provide a focused analysis of {ticker} using only those selected indicators:
   - Why you chose these specific indicators for {ticker}
   - What each indicator is currently showing for {ticker}
   - How they relate to {ticker}'s sector/size characteristics
   - Key support and resistance levels for {ticker} (derived from the data)
   - Any notable patterns or divergences in {ticker}
   - Near-term technical outlook for {ticker}

DO NOT provide buy/sell recommendations. Focus on objective technical observations about {ticker}.

**VERIFY: Before submitting, confirm every reference is to {ticker} ({stock_name}).**

Keep the analysis concise and actionable - about 300-400 words."""

        return PromptTemplate.from_template(template)
    
    def batch_analyze(self, tickers: List[str], analysis_type: str = "comprehensive") -> Dict[str, str]:
        """Analyze multiple tickers"""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.analyze(ticker, analysis_type)
            except Exception as e:
                results[ticker] = f"Error analyzing {ticker}: {str(e)}"
        return results


# Example usage
if __name__ == "__main__":
    agent = TechnicalAnalyzerAgent()
    
    print("=== Structured Output (for_synthesis) ===")
    print(agent.analyze("HOOD", analysis_type="for_synthesis"))
    
    print("\n" + "="*80 + "\n")
    
    print("=== Comprehensive Analysis ===")
    print(agent.analyze("HOOD", analysis_type="comprehensive"))